import os
import dgl
import random
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import pickle as pkl
from pathlib import Path
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix


# ==================== aux function ===============================
# in order to accelerate data processing speed,
# we save/load pickle files for SpeciesDGLDataset after construction
# NOTICE: pkl.dump location should be the samve with load location
#         otherwise, it would come across 'No Module Named Dataset' ERROR
def save_as_pickle(obj, file_name):
    file = open(file_name, 'wb')
    pkl.dump(obj, file)
    print('[SUCC] obj saved as {}'.format(file_name))


def load_from_pickle(file_name):
    file = open(file_name, 'rb')
    obj = pkl.load(file)
    print('[SUCC] obj get from {}'.format(file_name))
    return obj


# ========================================== three data-preprocessing classes ===========================================
# BIPARTITE-GRAPH  the target of 3 following 3 classes are aimed to construct one bipartite graph
#                  src_nodes are genes, tgt_nodes are cells
#                  edges are weighted and its biological meaning is expressive intensity of gene[i] on cell[j]
#                  edges are constructed to be bidirected on purpose for message passsing
# GRAPH-EXAMPLE                   1.07
#                     GENE O <------------> O CELL
#                          O <------------> O
#                                 1.22
# GENE-INFORMATION genes are not labeled, they comes from different species and different organs,
#                  different species (includes human, mouse) and different organs (includes lung, kidney) share genes
# CELL-INFORMATION cells are labels, their label means cell types (AT Cell, T Cell ...)
#                  different species (includes human, mouse) and different organs (includes lung, kidney) do not share cells
#                  different cells are independent in species and organs, we simple combines them together
# CLASS-RELATION   FileNxDataset (init input for SpeciesNxDataset/SpeciesDGLDataset)
#                  SpeciesNxDataset (father class [networkx graph data])
#                  SpceisDGLDataset (subclass [DGL graph data])


class FileNxDataset(object):
    """Used for constructing one obj to get a networkx graph from one file

       Attributes:
           G: networkx we get from one pair of (celltype and data) file
              graph is directed
              nodes have two attributes
              (
               bipartite = 0, 1  [0 means gene, 1 means cell]
               type_name ='10002','T Cell', 'AT Cell'...  [type_name can be gene_name_string or cell_name_string]
              )
              directed edges have one attributes
              (
               weight = 1.03, 2.11...
              )

       PS: isolated nodes would be automatically deleted when constructing bipartite networkx graph
           networkx graph can be visualized using function 'print_nx_graph'
    """

    def __init__(self, species, tissue, number, data_path):
        # put species, tissue, number into one tuple for easy param passing
        self.target_file_tuple = (species, tissue, str(number))
        self.G = nx.DiGraph()

        # collect location of *_data.csv and *_celltype.csv
        cell_type_file, data_file = self.get_file_path(data_path)

        # add gene nodes
        src_list = self.get_src(data_file)
        self.add_src_nodes(src_list, cell_type_file, data_file)

        # add cell nodes (cell has type attribute called 'type_name')
        tgt_list = self.get_tgt(cell_type_file)
        self.add_tgt_nodes(tgt_list, cell_type_file, data_file)

        # add edges between gene and cell with weight in form of numpy array (weight > 0)
        weight_mat = self.get_weight(data_file)
        self.add_edges_with_weight(src_list, tgt_list, weight_mat)

        self.G = self.delete_isolated_nodes()

    def get_file_path(self, data_path):
        species, tissue, number = self.target_file_tuple
        celltype_file_name = f'{species}_{tissue}{number}_celltype.csv'
        data_file_name = f'{species}_{tissue}{number}_data.csv'
        celltype_file = data_path / celltype_file_name
        data_file = data_path / data_file_name
        return celltype_file, data_file

    def get_tgt(self, cell_type_file):
        species, tissue, number = self.target_file_tuple
        cell_name_df = pd.read_csv(cell_type_file, dtype=str, usecols=[0])
        cell_name_list = cell_name_df.values[:, 0].tolist()
        cell_name_list = [
            species + '_' + tissue + '_' + number + '_' + cell_name
            for cell_name in cell_name_list
        ]
        return cell_name_list

    def get_src(self, data_file):
        gene_name_df = pd.read_csv(data_file, dtype=str, usecols=[0])
        gene_name_list = gene_name_df.values[:, 0].tolist()
        return gene_name_list

    def get_weight(self, data_file):
        gene_cell_mat_df = pd.read_csv(data_file)
        gene_cell_mat = gene_cell_mat_df.values[:, 1:]
        return gene_cell_mat

    def add_src_nodes(self, src_list, cell_type_file, data_file):
        self.G.add_nodes_from(src_list, bipartite=0)

    def add_tgt_nodes(self, tgt_list, cell_type_file, data_file):
        self.G.add_nodes_from(tgt_list, bipartite=1)
        # add cell_type attribute
        species, tissue, number = self.target_file_tuple
        cell_df = pd.read_csv(cell_type_file, usecols=[0, 1])
        # change Cell | Cell_type two columsn of csv into {Cell: Cell_type} dict
        cell_dict = cell_df.set_index('Cell').T.to_dict('records')[0]
        # add 'type_name' attributes based on 'Cell_type' in the dict
        for key, value in cell_dict.items():
            self.G.nodes[species + '_' + tissue + '_' + number + '_' +
                         key]['type_name'] = value

    def add_edges_with_weight(self, src_list, tgt_list, weight_mat):
        for i in range(len(src_list)):
            for j in range(len(tgt_list)):
                # if weight == 0, we ignore this edge
                if (weight_mat[i][j] > 0):
                    self.G.add_edge(src_list[i],
                                    tgt_list[j],
                                    weight=weight_mat[i][j])
                    # if bidirected, we need to add this line, else not
                    # but now we make converted DGLGraph to be bidirected, we have no need to make netowrkx graph bidirected anymore
                    #self.G.add_edge(tgt_list[j], src_list[i], weight=weight_mat[i][j])

    def delete_isolated_nodes(self):
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        return self.G

    def print_nx_graph(self):
        # Separate by group
        l, r = nx.bipartite.sets(self.G)
        pos = {}
        # Update position for node from each group
        pos.update((node, (1, index)) for index, node in enumerate(l))
        pos.update((node, (2, index)) for index, node in enumerate(r))
        nx.draw(self.G, with_labels=True, pos=pos)
        plt.show()


class SpeciesNxDataset(object):
    """Used for combining different network graphs into one large graph containing all information of one species
       This class also provide nx_to_dgl interface to change networkx into DGLGraph, however, DGLGraph not stored as attributes in this class
       The combining process means that we simple union all different cells into one new set {A,B} {C} ==> {A,B,C} since each cell is different
       When it comes to gene combination, we consider there do exist shared gene, we share shared genes, {1,2} {2,3} ==> {1,2,3}

       As a result, we need to feed a list to obj like SpeceisNxDataset([graph1, graph2]) in order to init this obj and start combining process

       Attributes:
           G: networkx containing all genes of one species (shared genes, special genes in different files)
                                  all cells of one species (all types in differnt organs and in different files)
                                  all edges between cells and genes which is not 0
              nodes with attributes 'bipartite', 'embed', 'type_name'(genes do not have)
              edges with attributes 'weight'

           node_index_dict: key = gene or cell string name, value = its index starting from 0  [e.g.{"C_1": 0, "10001": 1}]
           type_name_dict: key = cell string name, value = its string celltype name [e.g.{"C_1" : "T Cell"}]
           type_index_dict: key = cell string name, value = the index of its celltype name [e.g. C_1 and C_2 are both T Cell and C_3 is AT Cell, {"C_1" : 1, "C_2" : 1, "C_3" : 2}]
           gene_name_dict: key = gene string name, value = its index starting from 0 [e.g.{"10001": 0}]
           cell_name_dict: key = cell string name, value = its index starting from 0 [e.g.{"C_1": 0}]
           gene_num: num of src nodes in bipartite graph
           cell_num: num of dst nodes in bipartite graph

       PS: 'embed' form can be changed in get_gene_embed and get_cell_embed
           Here 'embed' means nodes features in GNN
           Although cells and genes are different types of nodes, we still consider the whole graph as homo-graph
           Although we do not store dgl graph in this class, we provide nx_to_dgl function for subclass SpeciesDGLDataset to get self.G into DGLGraph
    """

    def __init__(self, filebigraph_list):
        graph_list = [graph.G for graph in filebigraph_list]
        self.G = nx.compose_all(graph_list)
        self.node_index_dict = self.save_node_index_dict()
        self.type_name_dict = self.save_type_name_dict()
        self.type_index_dict = self.name_to_index(self.type_name_dict)
        self.gene_name_dict, self.cell_name_dict = self.save_gene_cell_name_dict(
        )
        self.gene_num, self.cell_num = self.check_gene_cell_num()
        self.get_gene_embed()
        self.get_cell_embed()

    def save_node_index_dict(self):
        # node_index_dict : {"C_1": 1, "10001": 2}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        nx_node_name_list = list(gene_set) + list(cell_set)

        node_index_dict = {}
        for i in range(len(nx_node_name_list)):
            node_index_dict[nx_node_name_list[i]] = i
        return node_index_dict

    def save_type_name_dict(self):
        # type_name_dict : {"C_1" : "T Cell"}
        type_name_dict = nx.get_node_attributes(self.G, "type_name")
        return type_name_dict

    def save_gene_cell_name_dict(self):
        # seperately store gene_name dict and cell_name dict for further use
        # two dict : {"C_1": 0}, {"10001": 1}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        gene_name_dict = {
            gene_name: idx for idx, gene_name in enumerate(list(gene_set))
        }
        cell_name_dict = {
            cell_name: idx for idx, cell_name in enumerate(list(cell_set))
        }
        return gene_name_dict, cell_name_dict

    def name_to_index(self, type_name_dict):
        type_index_dict = {}
        type_set = set()
        for key, value in type_name_dict.items():
            type_set |= {value}
        type_list = list(type_set)
        for key, value in type_name_dict.items():
            type_index_dict[key] = type_list.index(value)
        return type_index_dict

    def check_gene_cell_num(self):
        gene_num = len(self.gene_name_dict)
        cell_num = len(self.cell_name_dict)
        return gene_num, cell_num

    def get_gene_embed(self):
        for n in self.gene_name_dict:
            gene_embed = torch.zeros(self.gene_num)
            gene_embed[self.gene_name_dict[n]] = 1
            self.G.nodes[n]['embed'] = gene_embed

    def get_cell_embed(self):
        cell_embed = {}
        for n in self.cell_name_dict:
            cell_embed = torch.zeros(self.gene_num)
            for m in self.gene_name_dict:
                if self.G.get_edge_data(
                        m, n, default=0) and self.G[m][n]['weight'] > 0:
                    cell_embed[self.gene_name_dict[m]] = 1
            self.G.nodes[n]['embed'] = cell_embed

    def nx_to_dgl(self):
        dgl_G = dgl.from_networkx(self.G,
                                  edge_attrs=['weight'],
                                  node_attrs=['embed'])
        dgl_G = self.make_bidirected(dgl_G)
        return dgl_G

    def make_bidirected(self, dgl_G):
        src_node_tensor = dgl_G.edges()[0]
        tgt_node_tensor = dgl_G.edges()[1]
        weight_tensor = dgl_G.edata['weight']
        dgl_G.add_edges(tgt_node_tensor, src_node_tensor)
        return dgl_G

    def print_embed(self):
        print(self.gene_name_dict)
        for n, _ in self.G.nodes(data=True):
            print(n, self.G.nodes[n]['embed'])

    def print_nx_graph(self):
        # separate by group
        l, r = nx.bipartite.sets(self.G)
        pos = {}
        # update position for node from each group
        pos.update((node, (1, index)) for index, node in enumerate(l))
        pos.update((node, (2, index)) for index, node in enumerate(r))
        nx.draw(self.G, with_labels=True, pos=pos)
        plt.show()


class SpeciesDGLDataset(SpeciesNxDataset):
    """Use for changing Networkx Graph into DGL graph (inherited from SpeciesNxDataset)
       After this transformation, we finally reach the final point of data preprocessing
       and get the final Bipartite DGLGraph represents all genes and cells and edges in one species.

       As a result, we need to init one SpeciesDGLDataset based on nx_data in SpeciesNxDataset and dict in SpeciesNxDataset.
       We need to feed a list to obj like SpeceisNxDataset([graph1, graph2]) in order to init this obj and start combining process.

       Attributes:
           G: DGLGraph converted from networkx graph (even though during the transformation DGL by default would become directed, we make DGLGraph bidirected again)
              nodes with attributes 'embed' (tensor version)
              edges with attributes 'weight' (tensor version)
              [DGLGraph does not support non-tensor attributes]
           feature: embed of DGLGraph nodes, we get it and form one matrix
           label: use type_index_dict in Class SpeciesNxDataset {"C_1" : 1, "C_2" : 1, "C_3" : 2} to get numerical labels for all cells, genes do not have labels
           train/valid/test_index: split [1,2,3,4,5] into [1,2,3], [4], [5], each is a list

        PS: load_data function can output all necessary information for model training
    """

    def __init__(self, filebigraph_list):
        super(SpeciesDGLDataset, self).__init__(filebigraph_list)
        self.G = self.nx_to_dgl()
        self.feature = self.extract_feature()
        self.label = self.extract_label()
        self.train_index, self.valid_index, self.test_index = self.split_cell_nodes(
        )

    def extract_feature(self):
        # we need to change 'embed' into csr_matrix since it is too large to successfully pickle
        feature_tensor = csr_matrix(self.G.ndata['embed'])
        # delete embed from the self.G since embed matrix is too big more than (10000 * 10000)
        # thus, after getting feature_tensor, we delete ndata['embed']
        self.G.ndata.pop('embed')
        return feature_tensor

    def extract_label(self):
        label_list = []
        for key, _ in self.node_index_dict.items():
            if key in self.type_index_dict.keys():
                label_list += [self.type_index_dict[key]]
            else:
                label_list += [-1]
        label_tensor = torch.tensor(label_list)
        return label_tensor

    def split_cell_nodes(self):
        # only calculate loss on cells
        # TODO: here we suppose gene name is smaller than cell name
        cell_dataset = self.G.nodes()[self.gene_num:]
        train_subset, valid_subset, test_subset = dgl.data.utils.split_dataset(
            cell_dataset, shuffle=True, frac_list=[0.7, 0.2, 0.1])
        train_index = train_subset[:]
        valid_index = valid_subset[:]
        test_index = test_subset[:]
        return train_index, valid_index, test_index

    def load_data(self):
        return self.G, self.feature, self.label, self.train_index, self.valid_index, self.test_index

    def print_dgl_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.G.to_networkx(), ax=ax)
        plt.show()
