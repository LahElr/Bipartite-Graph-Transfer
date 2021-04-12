import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np
from utils import lahelr_print
"""
    Graph Transformer Layer with edge features
    
"""
"""
    Util functions
"""


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}

    return func


# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
        # lahelr:
        # shape e,8,4,4 * shape e,[],4,4
        # What we want is actually the mult of :
        # feature of every node(vector,length 8, for example)(processed)
        #  and 
        # the feature of every edge(scaler, for example)(processed)
        # so we should unsqueeze the scaler part to the size of node feature

        # eg. 3,x -> 3,2,x

        # [x1,x2,x3] ->
        # [[x1,x1]
        #  [x2,x2]
        #  [x3,x3]]

    """
    def func(edges):
        return {
            implicit_attn:
            (edges.data[implicit_attn] * edges.data[explicit_edge])
        }

    return func


# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}

    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {
            field:
            torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))
        }

    return func


"""
    Single Attention Head
"""


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, feat_size, weight_size,
                 use_bias):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.feat_size = feat_size
        self.weight_size = weight_size

        lahelr_print("in_dim and out_dim:", in_dim, out_dim, "*", num_heads)

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):

        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  #, edges)
        # 'score': (edges.src['K_h'] * edges.dst['Q_h'])    shape(edge_num, feat_size, edge_size, num_heads, out_dim)   e,8,4,4

        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        # 'score': ((edges.data['score']) / np.sqrt(self.out_dim))

        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # "score": (edges.data['score'] * edges.data['proj_e'].unsqueeze(1).repeat(1,8,1,1))     shape e,8,4,4 * shape e,8,4,4

        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        # 'e_out': edges.data['score']

        # softmax
        g.apply_edges(exp('score'))
        # "score": torch.exp((edges.data['score'].sum(-1, keepdim=True)).clamp(-5, 5))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'),
                        fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'),
                        fn.sum('score', 'z'))

    def forward(self, g, h, e):

        lahelr_print("g,h,e when in the attention layer:", g, h.shape, e.shape)

        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        lahelr_print("Q_h,K_h,V_h,proj_e :", Q_h.shape, K_h.shape, V_h.shape,
                     proj_e.shape)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        # g.ndata['Q_h'] = Q_h.view(-1,*self.feat_size,self.num_heads,self.out_dim)
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)

        lahelr_print("Q_h,K_h,V_h,proj_e before propagate_attention:",
                     g.ndata['Q_h'].shape, g.ndata['K_h'].shape,
                     g.ndata['V_h'].shape, g.edata['proj_e'].shape)

        self.propagate_attention(g)

        _ = lambda x:(x[0],x[1].shape)
        lahelr_print("after propagate_attention:")
        lahelr_print(*[x for x in map(_,g.ndata.items())],sep=', ')
        lahelr_print(*[x for x in map(_,g.edata.items())],sep=', ')

        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(
            g.ndata['z'], 1e-6))  # adding eps to all values here
        e_out = g.edata['e_out']

        lahelr_print("attention layer returns: h_out and e_out",h_out.shape,e_out.shape)

        return h_out, e_out


class WeightedGraphTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_size,
                 weight_size,
                 dropout=0.0,
                 layer_norm=False,
                 batch_norm=True,
                 residual=True,
                 use_bias=False):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.feat_size = feat_size
        self.weight_size = weight_size

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads,
                                                 num_heads, self.feat_size,
                                                 self.weight_size, use_bias)

        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            self.layer_norm2_e = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            self.batch_norm2_e = nn.BatchNorm1d(out_dim)

    def forward(self, g, h, e):
        lahelr_print("g,h,e when in the transformer layer:", g, h.shape,
                     e.shape)  # 13 4, 25 4
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection

        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)

        lahelr_print("g,h,e when in the transformer layer, after attention layer:", g, h.shape,
                     e.shape)

        # h = h_attn_out.view(-1, *self.feat_size, self.out_channels)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            e = self.batch_norm1_e(e)

        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        lahelr_print("g,h,e when in the transformer layer, after 2nd residual:", g, h.shape,
                     e.shape)

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        lahelr_print("g,h,e when in the transformer layer, after FEN:", g, h.shape,
                     e.shape)

        if self.residual:
            h = h_in2 + h  # residual connection
            e = e_in2 + e  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)
            e = self.layer_norm2_e(e)

        if self.batch_norm:
            h = self.batch_norm2_h(h)
            e = self.batch_norm2_e(e)

        lahelr_print("g,h,e that transformer layer returns", g, h.shape,
                     e.shape)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_heads, self.residual)
