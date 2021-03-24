import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn.pytorch.conv import SAGEConv
from torch.utils.data import DataLoader
from dgl.utils import expand_as_pair, check_eq_shape


class NeighborSampler(object):
    """Used for NeighborSample used in minibatch-GraphSAGE process
    sample_blocks function is used for DataLoader to help create batches of data

    Attributes:
        g: DGLGraph to sample
        fanouts: list of number, if [10, 20] means first layer sampling 10 nodes (1-hop neighbor)
                 second layer sampling 20 nodes (2-hop neighbor)
    """

    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        # seeds --> minibatch nodes
        # NOTICE seeds[0] is init input
        #        frontier[0] = [seeds, sampled neighbors]
        #        seeds[1] = frontier[0]
        #        Thus, seed[0] do sample again in the 2nd iteration

        # sample_blocks is used for DataLoader.collate_fn
        # seeds provided by DataLoader (actually minibatch nodes)
        seeds = torch.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # (2-hop sampled neighbor) ------------ (1-hop sampled neighbor) ---------(minibatch nodes)
            # sample_neighbors means seeds as minibatch nodes, sample their neighbors
            # replace=True means view sampled neighbor as the whole neighbor of minibatch nodes
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # change sampled neighbor and minibatch nodes into bipartite graph 
            # for convenient of message passing
            # block.srcdata (sampled neighbor + minibatch nodes)
            # block.dstdata (minibatch nodes)
            block = dgl.to_block(frontier, seeds)
            # change neighbors to be minibatch nodes in order to prepare for the next layer sampling
            seeds = block.srcdata[dgl.NID]
            # len(blocks) are k-hop of sampling
            blocks.insert(0, block)
        return blocks



class GraphSAGE(nn.Module):
    """Baseline Model 1 ( GraphSAGE )
    forward with blocks ( each block is one bipartite graph ) and x ( feature matrix in one batch )

    Attributes:
        n_layers: choose k-hop of neighbors to be sampled
        n_hidden: hidden states size for SAGEConv layers, we only provide one type of layer size
        n_classes: celltype classes, not include gene types (since supervise signal only put on cell types)
        layers: SAGEConv layers with [in_feats, n_hidden, n_hidden, ..., n_hidden, n_classes]
        dropout, activation: ...
    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist
        self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.fc = nn.Linear(in_feats, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        # blocks is list of block
        # block is the bipartite graph we sample. Here it is used for message passing.
        # x is node feature, includes [dst_nodes, src_nodes] in x, dst_nodes are mini-batch nodes and src_nodes are sampled neighbors
        # e.g. mini-batch [1,2,3], their neighbors [4,5,6], h should be hidden states for [1,2,3,4,5,6]
        h = self.fc(x)
        h = self.activation(h)
        h = self.dropout(h)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # get minibatch nodes hidden state
            h_dst = h[:block.number_of_dst_nodes()]
            # SAGEConv input is quite special
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class WeightedSAGEConv(nn.Module):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(WeightedSAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def edge_message(self, edges):
        # edges.src with the shape of (#edges, embed_dim)
        # edges.data['weight'] with the shape of (#edges, 1)
        # broadcast in order to update with weight
        w = edges.data['weight'].float().reshape(-1, 1)
        return {'m': edges.src['h'] * w}
    
    def forward(self, graph, feat):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            graph.update_all(self.edge_message, fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst     # same as above if homogeneous
            graph.update_all(self.edge_message, fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(self.edge_message, fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(self.edge_message, self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class WeightedGraphSAGE(nn.Module):
    """Baseline Model 1 ( GraphSAGE )
    forward with blocks ( each block is one bipartite graph ) and x ( feature matrix in one batch )

    Attributes:
        n_layers: choose k-hop of neighbors to be sampled
        n_hidden: hidden states size for SAGEConv layers, we only provide one type of layer size
        n_classes: celltype classes, not include gene types (since supervise signal only put on cell types)
        layers: SAGEConv layers with [in_feats, n_hidden, n_hidden, ..., n_hidden, n_classes]
        dropout, activation: ...
    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist
        self.layers.append(WeightedSAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(WeightedSAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(WeightedSAGEConv(n_hidden, n_classes, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        # blocks is list of block
        # block is the bipartite graph we sample. Here it is used for message passing.
        # x is node feature, includes [dst_nodes, src_nodes] in x, dst_nodes are mini-batch nodes and src_nodes are sampled neighbors
        # e.g. mini-batch [1,2,3], their neighbors [4,5,6], h should be hidden states for [1,2,3,4,5,6]
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # get minibatch nodes hidden state
            h_dst = h[:block.number_of_dst_nodes()]
            # SAGEConv input is quite special
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

