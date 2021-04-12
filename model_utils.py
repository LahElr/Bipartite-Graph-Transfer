import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair, check_eq_shape
from bidict import bidict
import pickle

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
        seeds_list = [seeds]
        fanout_when_0 = max(self.fanouts)
        for fanout in self.fanouts:
            # (2-hop sampled neighbor) ------------ (1-hop sampled neighbor) ---------(minibatch nodes)
            # sample_neighbors means seeds as minibatch nodes, sample their neighbors
            # replace=True means view sampled neighbor as the whole neighbor of minibatch nodes
            # the frontier contians the sampled subgraph of neighbors of nodes in `seeds`
            # TODO if the fanout is 0, it means we want all nodes' neighbors to be involved into the next sample,
            # TODO in this condition, we sample `fanout_when_0` neighbors and add them th the blocks
            frontier = dgl.sampling.sample_neighbors(
                self.g, seeds, fanout if fanout != 0 else fanout_when_0, replace=False)
            # change sampled neighbor and minibatch nodes into bipartite graph
            # for convenient of message passing
            # block.srcdata (sampled neighbor + minibatch nodes)
            # block.dstdata (minibatch nodes)
            # ! convert the frontier to the block, a bipartite graph
            block = dgl.to_block(frontier, seeds)
            # change neighbors to be minibatch nodes in order to prepare for the next layer sampling
            # ! update the nodes to be sample-neighbor to the now neighbors
            seeds = block.srcdata[dgl.NID]
            seeds_list.append(seeds)
            # len(blocks) are k-hop of sampling
            # // graph = dgl.block_to_graph(block) # this func is not usable
            blocks.insert(0, block)

        graph, node_map, edge_map = self.convert_blocks_to_graph(blocks)

        return (graph, node_map, edge_map)
    
    def convert_blocks_to_graph(self, blocks):
        # TODO if the fanout is 0, it means we want all nodes' neighbors to be involved into the next sample,
        # TODO in this condition, we sample `fanout_when_0` neighbors and add them th the blocks
        # convert the blocks into a DGLGraph
        # and make the labels to be returned together with the graph
        # create a new graph to contain these sampled nodes, set the features and weights in,
        # and make a dict to record the mapping of original node IDs and node IDs in this graph
        # we can use bidict
        # graph = dgl.DGLGraph()
        graph = dgl.graph([])
        node_map = bidict()  # node id in new graph -> node id in original graph
        edge_map = bidict()

        # add all nodes sampled to the graph
        node_ct = 0
        for block in blocks:
            for node_id_in_original_graph in block.srcdata[dgl.NID]:
                node_id_in_original_graph = int(node_id_in_original_graph)
                if node_map.inverse.get(node_id_in_original_graph, None) is None:
                    node_id_in_new_graph = node_ct
                    node_ct+=1
                    node_map[node_id_in_new_graph] = node_id_in_original_graph
        graph.add_nodes(node_ct)


        # add all node data to the graph
        for node_id_in_new_graph in range(graph.number_of_nodes()):
            node_id_in_original_graph = node_map[node_id_in_new_graph]
            for key in self.g.ndata.keys():
                try:
                    graph.ndata[key][node_id_in_new_graph] = self.g.ndata[key][node_id_in_original_graph]
                except KeyError:
                    shape = list(self.g.ndata[key].shape)
                    shape[0] = graph.number_of_nodes()
                    shape = tuple(shape)
                    graph.ndata[key] = torch.zeros(
                        shape, dtype=self.g.ndata[key].dtype)
                    graph.ndata[key][node_id_in_new_graph] = self.g.ndata[key][node_id_in_original_graph]

        # add all edges into the graph
        for block in blocks:
            for edge_id_in_block in range(block.number_of_edges()):
                # edge_id_in_original_graph = int(block.edata[dgl.EID][edge_id_in_block])
                start_node_id_original = node_map.get(
                    block.find_edges(edge_id_in_block)[0].numpy()[0], None)
                end_node_id_original = node_map.get(
                    block.find_edges(edge_id_in_block)[1].numpy()[0], None)
                edge_id_in_original_graph_s = self.g.edge_ids(
                    start_node_id_original, end_node_id_original)
                if isinstance(edge_id_in_original_graph_s, (int, float)):
                    edge_id_in_original_graph_s = [
                        int(edge_id_in_original_graph_s)]
                elif isinstance(edge_id_in_original_graph_s, (torch.Tensor)):
                    edge_id_in_original_graph_s = map(
                        int, list(edge_id_in_original_graph_s.numpy()))
                for edge_id_in_original_graph in edge_id_in_original_graph_s:
                    if edge_map.inverse.get(edge_id_in_original_graph, None) is None:
                        graph.add_edges(node_map.inverse[start_node_id_original],
                                       node_map.inverse[end_node_id_original])
                        edge_id_in_new_graph = graph.number_of_edges()-1
                        edge_map[edge_id_in_new_graph] = edge_id_in_original_graph

        # add all edge feature into the graph
        for edge_id_in_new_graph in range(graph.number_of_edges()):
            edge_id_in_original_graph = edge_map.get(
                edge_id_in_new_graph, None)
            for key in self.g.edata.keys():
                try:
                    graph.edata[key][edge_id_in_new_graph] = self.g.edata[key][edge_id_in_original_graph]
                except KeyError:
                    shape = list(self.g.edata[key].shape)
                    shape[0] = graph.number_of_edges()
                    shape = tuple(shape)
                    graph.edata[key] = torch.zeros(
                        shape, dtype=self.g.edata[key].dtype)
                    graph.edata[key][edge_id_in_new_graph] = self.g.edata[key][edge_id_in_original_graph]

        return graph, node_map, edge_map


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
            self.lstm = nn.LSTM(self._in_src_feats,
                                self._in_src_feats,
                                batch_first=True)
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
        m = nodes.mailbox['m']  # (B, L, D)
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
            graph.dstdata['h'] = feat_dst  # same as above if homogeneous
            graph.update_all(self.edge_message, fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata['neigh'] +
                       graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(self.edge_message, fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(self.edge_message, self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(
                self._aggre_type))

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
