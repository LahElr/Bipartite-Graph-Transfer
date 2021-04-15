import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn.pytorch.conv import SAGEConv
from torch.utils.data import DataLoader
from model_utils import WeightedSAGEConv

from bidict import bidict
import pickle
from utils import lahelr_print
from layers.graph_transformer_edge_layer import WeightedGraphTransformerLayer
from layers.graph_transformer_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout


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

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(
            self.fc.weight, gain=nn.init.calculate_gain('relu'))

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
        return self.fc(h)


class WeightedGraphSAGE(nn.Module):
    """Baseline Model 2 ( WeightedGraphSAGE )
    forward with blocks ( each block is one bipartite graph ) and x ( feature matrix in one batch )

    Attributes:
        n_layers: choose k-hop of neighbors to be sampled
        n_hidden: hidden states size for SAGEConv layers, we only provide one type of layer size
        n_classes: celltype classes, not include gene types (since supervise signal only put on cell types)
        layers: SAGEConv layers with [in_feats, n_hidden, n_hidden, ..., n_hidden, n_classes]
        dropout, activation: ...
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 feat_drop):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist

        # if one hip sampling, only one layer
        self.layers.append(WeightedSAGEConv(
            in_feats, n_hidden, 'gcn', activation=activation, feat_drop=feat_drop))
        # if more than one hip sampling, more than one layer SAGEConv
        for i in range(n_layers - 1):
            self.layers.append(WeightedSAGEConv(
                n_hidden, n_hidden, 'gcn', activation=activation, feat_drop=feat_drop))
        self.fc = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(
            self.fc.weight, gain=nn.init.calculate_gain('relu'))
        #self.out = WeightedSAGEConv(n_hidden, n_classes, 'gcn', activation=activation, feat_drop=feat_drop)

    def forward(self, blocks, x):
        # blocks is list of block
        # block is the bipartite graph we sample. Here it is used for message passing.
        # x is node feature, includes [dst_nodes, src_nodes] in x, dst_nodes are mini-batch nodes and src_nodes are sampled neighbors
        # e.g. mini-batch [1,2,3], their neighbors [4,5,6], h should be hidden states for [1,2,3,4,5,6]
        h = x

        # len(blocks means the hop of graph sampling)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # get minibatch nodes hidden state
            h_dst = h[:block.number_of_dst_nodes()]
            # SAGEConv input is quite special
            h = layer(block, (h, h_dst))
        return self.fc(h)


class GraphSAGEwithEmbed(nn.Module):
    """Baseline Model 3 ( GraphSAGE with one embedding layer )
    forward with blocks ( each block is one bipartite graph ) and x ( feature matrix in one batch )

    Attributes:
        n_layers: choose k-hop of neighbors to be sampled
        n_hidden: hidden states size for SAGEConv layers, we only provide one type of layer size
        n_classes: celltype classes, not include gene types (since supervise signal only put on cell types)
        layers: SAGEConv layers with [in_feats, n_hidden, n_hidden, ..., n_hidden, n_classes]
        dropout, activation: ...
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist
        self.embed = nn.Linear(in_feats, n_hidden, bias=False)
        for i in range(n_layers):
            self.layers.append(SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(
            self.fc.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, blocks, x):
        # blocks is list of block
        # block is the bipartite graph we sample. Here it is used for message passing.
        # x is node feature, includes [dst_nodes, src_nodes] in x, dst_nodes are mini-batch nodes and src_nodes are sampled neighbors
        # e.g. mini-batch [1,2,3], their neighbors [4,5,6], h should be hidden states for [1,2,3,4,5,6]
        h = self.embed(x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # get minibatch nodes hidden state
            h_dst = h[:block.number_of_dst_nodes()]
            # SAGEConv input is quite special
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return self.fc(h)


class WeightedGraphSAGEwithEmbed(nn.Module):
    """Baseline Model 4 ( GraphSAGE with one embedding layer )
    forward with blocks ( each block is one bipartite graph ) and x ( feature matrix in one batch )

    Attributes:
        n_layers: choose k-hop of neighbors to be sampled
        n_hidden: hidden states size for SAGEConv layers, we only provide one type of layer size
        n_classes: celltype classes, not include gene types (since supervise signal only put on cell types)
        layers: SAGEConv layers with [in_feats, n_hidden, n_hidden, ..., n_hidden, n_classes]
        dropout, activation: ...
    """

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        # add layers into Modulelist
        self.embed = nn.Linear(in_feats, n_hidden, bias=False)
        for i in range(n_layers):
            self.layers.append(WeightedSAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(WeightedSAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.fc = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(
            self.fc.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, blocks, x):
        # blocks is list of block
        # block is the bipartite graph we sample. Here it is used for message passing.
        # x is node feature, includes [dst_nodes, src_nodes] in x, dst_nodes are mini-batch nodes and src_nodes are sampled neighbors
        # e.g. mini-batch [1,2,3], their neighbors [4,5,6], h should be hidden states for [1,2,3,4,5,6]
        h = self.embed(x)
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # get minibatch nodes hidden state
            h_dst = h[:block.number_of_dst_nodes()]
            # SAGEConv input is quite special
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return self.fc(h)


class WeightedGraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        # node_dim (feat is an integer)
        in_dim_node = net_params['in_dim_node']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.readout = net_params['readout']
        self.edge_feat = net_params['edge_feat']
        self.feat_size = net_params['feat_size']
        self.weight_size = net_params['weight_size']
        self.weight_pre = net_params['weight_pre']
        self.feat_pre = net_params['feat_pre']
        max_wl_role_index = net_params["max_wl_role_index"]

        # if the positional encoding is applied
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index,
                                                     hidden_dim)
        if self.feat_pre == 'embedding':
            print("warning: do you sure the node feature is integer?")
            self.embedding_h = nn.Embedding(
                in_dim_node, hidden_dim)  # node feat is an integer
        elif self.feat_pre == 'linear':
            # feat_size -> feat_size[:-1] || hidden_dim
            # feat_size[-1] = in_dim_node
            self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        else:
            print("warning: do you sure the node feature is integer?")
            self.embedding_h = nn.Embedding(
                in_dim_node, hidden_dim)  # node feat is an integer

        if self.edge_feat:
            if self.weight_pre == 'embedding':
                self.embedding_e = nn.Embedding(in_dim_edge, hidden_dim)
            elif self.weight_pre == 'linear':
                self.embedding_e = nn.Linear(1, hidden_dim)
            else:
                self.embedding_e = nn.Embedding(in_dim_edge, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([WeightedGraphTransformerLayer(hidden_dim, hidden_dim, num_heads, self.feat_size, self.weight_size,
                                                                   dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(WeightedGraphTransformerLayer(
            hidden_dim, out_dim, num_heads, self.feat_size, self.weight_size, dropout, self.layer_norm, self.batch_norm,  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        lahelr_print("h after preprocess:", h.shape)  # 13 4

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.shape).to(self.device)
            e = self.embedding_e(e)
        else:
            if self.weight_pre == 'embedding':
                e = self.embedding_e(e)
            elif self.weight_pre == 'linear':
                e = self.embedding_e(e)
            else:
                e = self.embedding_e(e)
        lahelr_print("e after preprocess:", e.shape)  # 25 4

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == 'None' or self.readout is None:
            hg = h
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        lahelr_print("hg to be readout:", hg.shape)

        h_out = self.MLP_layer(hg)

        return h_out

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        lahelr_print("loss: the scores and targets:",
                     pred.shape, label.shape)
        lahelr_print(
            "loss: the pridicted labels and the true labels:", pred, label, sep='\n')
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim_node']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        self.readout = net_params['readout']
        self.edge_feat = net_params['edge_feat']
        self.feat_size = net_params['feat_size']
        self.weight_size = net_params['weight_size']
        self.weight_pre = net_params['weight_pre']
        self.feat_pre = net_params['feat_pre']
        max_wl_role_index = net_params["max_wl_role_index"]

        # if the positional encoding is applied
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index,
                                                     hidden_dim)

        if self.feat_pre == 'embedding':
            print("warning: do you sure the node feature is integer?")
            self.embedding_h = nn.Embedding(
                in_dim_node, hidden_dim)  # node feat is an integer
        elif self.feat_pre == 'linear':
            # feat_size -> feat_size[:-1] || hidden_dim
            # feat_size[-1] = in_dim_node
            self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        else:
            print("warning: do you sure the node feature is integer?")
            self.embedding_h = nn.Embedding(
                in_dim_node, hidden_dim)  # node feat is an integer

        # whether edge feature is applied
        if self.edge_feat:
            if self.weight_pre == 'embedding':
                self.embedding_e = nn.Embedding(in_dim_edge, hidden_dim)
            elif self.weight_pre == 'linear':
                self.embedding_e = nn.Linear(1, hidden_dim)
            else:
                self.embedding_e = nn.Embedding(in_dim_edge, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, hidden_dim, num_heads,
                                  self.feat_size, self.weight_size, dropout,
                                  self.layer_norm, self.batch_norm,
                                  self.residual) for _ in range(n_layers - 1)
        ])
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads,
                                  self.feat_size, self.weight_size, dropout,
                                  self.layer_norm, self.batch_norm,
                                  self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):
        '''
        g: the batched graph
        h: node feature
        e: edge feature
        h_lap_pos_enc: prepared laplacian positional encoding
        h_wl_pos_enc: prepared wl postional encoding

        positional encoding->emdedding
                                ↓                                   G -> update hidden node data to h -> readout -> MLPReadout ->
        h->embedding->dropout-> + -> [graph_transformer_layers]-> h ----->↗
             e(if there is)->embedding↗  ↘h,e↗..10..↘h,e↗     e -×
        '''

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        lahelr_print("h after preprocess:", h.shape)  # 13 4

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

        if not self.edge_feat:  # edge feature set to 1
            e = torch.ones(e.shape).to(self.device)
            e = self.embedding_e(e)
        else:
            if self.weight_pre == 'embedding':
                e = self.embedding_e(e)
            elif self.weight_pre == 'linear':
                e = self.embedding_e(e)
            else:
                e = self.embedding_e(e)
        lahelr_print("e after preprocess:", e.shape)  # 25 4

        # convnets
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == 'None' or self.readout is None:
            hg = h
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        lahelr_print("hg to be readout:", hg.shape)

        hg = self.MLP_layer(hg)

        lahelr_print("hg after readout:", hg.shape)

        return hg

    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        lahelr_print("loss: the scores and targets:",
                     scores.shape, targets.shape)
        lahelr_print(
            "loss: the pridicted labels and the true labels:", scores, targets, sep='\n')
        # loss = nn.L1Loss()(scores, targets)
        loss = nn.CrossEntropyLoss(scores, targets)
        return loss
