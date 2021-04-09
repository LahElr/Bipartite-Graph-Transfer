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
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))

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
        self.layers.append(WeightedSAGEConv(in_feats, n_hidden, 'gcn', activation=activation, feat_drop=feat_drop))
        # if more than one hip sampling, more than one layer SAGEConv
        for i in range(n_layers - 1):
            self.layers.append(WeightedSAGEConv(n_hidden, n_hidden, 'gcn', activation=activation, feat_drop=feat_drop))
        self.fc = nn.Linear(n_hidden, n_classes)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
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
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))

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

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
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
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))

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

