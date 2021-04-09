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
        
    def save_nodestr_nodeindex(self):
        # nodestr_nodeindex : {"C_1": 1, "10001": 2}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        nx_node_name_list = list(gene_set) + list(cell_set)

        nodestr_nodeindex = {}
        for i in range(len(nx_node_name_list)):
            nodestr_nodeindex[nx_node_name_list[i]] = i
        return nodestr_nodeindex

    def save_cellstr_typestr(self):
        # cellstr_typestr : {"C_1" : "T Cell"}
        cellstr_typestr = nx.get_node_attributes(self.G, "type_name")
        return cellstr_typestr

    def from_str_to_index(self):
        # cellstr_typeindex : {"C_1" : 1, "C_2" : 1, "C_3" : 2}
        # typestr_typeindex : {"T Cell" : 1, "AT Cell" : 2}
        cellstr_typeindex = {}
        typestr_typeindex = {}
        type_set = set()
        for key, value in self.cellstr_typestr.items():
            type_set |= {value}
        type_list = list(type_set)
        
        for idx, typestr in enumerate(type_list):
            typestr_typeindex[typestr] = idx
        for key, value in self.cellstr_typestr.items():
            cellstr_typeindex[key] = type_list.index(value)
            
        return cellstr_typeindex, typestr_typeindex
    
    def save_gene_cell(self):
        # seperately store gene_name dict and cell_name dict for further use
        # genestr_geneindex : {"C_1": 0}
        # cellstr_cellidnex : {"10001": 1}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        genestr_geneindex = {
            gene_name: idx for idx, gene_name in enumerate(list(gene_set))
        }
        cellstr_cellindex = {
            cell_name: idx for idx, cell_name in enumerate(list(cell_set))
        }
        return genestr_geneindex, cellstr_cellindex

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
        return torch.tensor(gene_cell_mat)

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
        row_idxes, col_idxes = torch.nonzero(weight_mat > 0, as_tuple=True)
        pos_weight_num = row_idxes.shape[0]
       
        edge_num = 0
        for idx in range(pos_weight_num):
            i = row_idxes[idx]
            j = col_idxes[idx]
            if weight_mat[i][j] > 0:
                edge_num += 1
                self.G.add_edge(src_list[i],
                                tgt_list[j],
                                weight=weight_mat[i][j])
                # if bidirected, we need to add this line, else not
                # but now we make converted DGLGraph to be bidirected, we have no need to make netowrkx graph bidirected anymore
                #self.G.add_edge(tgt_list[j], src_list[i], weight=weight_mat[i][j])
        print("edge_num : {}".format(edge_num))

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
           
           G': only choose #chosen_type_num types of cells sinice many types of cells have few samples,
               these types might cause data inbalance

           nodestr_nodeindex: key = gene or cell string name, value = its index starting from 0  [e.g.{"C_1": 0, "10001": 1}]
           cellstr_typestr: key = cell string name, value = its string celltype name [e.g.{"C_1" : "T Cell"}]
           cellstr_typeindex: key = cell string name, value = the index of its celltype name [e.g. C_1 and C_2 are both T Cell and C_3 is AT Cell, {"C_1" : 1, "C_2" : 1, "C_3" : 2}]
           typestr_typeindex: key = celltype name, value = its index [e.g. {"T Cell" : 1, "AT Cell" : 2}]
           genestr_geneindex: key = gene string name, value = its index starting from 0 [e.g.{"10001": 0}]
           cellstr_cellindex: key = cell string name, value = its index starting from 0 [e.g.{"C_1": 0}]
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
        
        #chosen_typestr = ['Dendritic cell', 'Stromal cell', 'T cell', 'AT2 cell', 'Endothelial cell']
        #chosen_typestr = ['AT Cell']
        #self.G = self.extract_certain_classes(chosen_typestr)
        
        chosen_type_num = 6
        self.G, chosen_typestr = self.extract_topk_classes_subgraph(chosen_type_num)
        
        # get bunch of dict for future usage
        self.nodestr_nodeindex = self.save_nodestr_nodeindex()
        self.cellstr_typestr = self.save_cellstr_typestr()
        self.cellstr_typeindex, self.typestr_typeindex = self.from_str_to_index()
        self.genestr_geneindex, self.cellstr_cellindex = self.save_gene_cell()
        self.nodeindex_geneindex, self.geneindex_nodeindex = self.index_convert()
        self.topkclassesstr_topkclassesindex = self.get_topktypedict(chosen_typestr)
        
        self.gene_num, self.cell_num = self.get_gene_cell_num()
        self.get_gene_embed()
        self.get_cell_embed()
        
    def delete_isolated_nodes(self):
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        return self.G
    
    def extract_certain_classes(self, chosen_typestr):
        deleted_cells = []
        
        # get deleted cell nodes
        for n, d in self.G.nodes(data=True):
            if d['bipartite'] == 1 and d['type_name'] not in chosen_typestr:
                deleted_cells.append(n)
            
        # remove cell nodes
        for n in deleted_cells:
            self.G.remove_node(n)
        # remove isolated nodes
        self.G = self.delete_isolated_nodes()
        return self.G
    
    def extract_topk_classes_subgraph(self, type_num=3):
        typestr_list = list(nx.get_node_attributes(self.G, "type_name").values())
        typestr_typenum = {}
        for key in typestr_list:
            typestr_typenum[key] = typestr_typenum.get(key,
                                 0) + 1  # dict value default 0, add 1 each time

        # choose class_num classes in labels, class_num classes has topk cells for labels
        chosen_typestr_tuples = sorted(typestr_typenum.items(),
                                     key=lambda x: x[1])[-type_num:]
        # chosen_typestr like ['T Cell', 'AT Cell']
        chosen_typestr = [dict_tuple[0] for dict_tuple in chosen_typestr_tuples]
        
        # delete nodes of not chosen typestr
        deleted_nodes = []
        for n, d in self.G.nodes(data=True):
            if d['bipartite'] == 1 and d['type_name'] not in chosen_typestr:
                deleted_nodes.append(n)
        for n in deleted_nodes:
            self.G.remove_node(n)
        self.G = self.delete_isolated_nodes()
        return self.G, chosen_typestr
        
    def save_nodestr_nodeindex(self):
        # nodestr_nodeindex : {"C_1": 1, "10001": 2}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        nx_node_name_list = list(gene_set) + list(cell_set)

        nodestr_nodeindex = {}
        for i in range(len(nx_node_name_list)):
            nodestr_nodeindex[nx_node_name_list[i]] = i
        return nodestr_nodeindex

    def save_cellstr_typestr(self):
        # cellstr_typestr : {"C_1" : "T Cell"}
        cellstr_typestr = nx.get_node_attributes(self.G, "type_name")
        return cellstr_typestr

    def from_str_to_index(self):
        # cellstr_typeindex : {"C_1" : 1, "C_2" : 1, "C_3" : 2}
        # typestr_typeindex : {"T Cell" : 1, "AT Cell" : 2}
        cellstr_typeindex = {}
        typestr_typeindex = {}
        type_set = set()
        for key, value in self.cellstr_typestr.items():
            type_set |= {value}
        type_list = list(type_set)
        
        for idx, typestr in enumerate(type_list):
            typestr_typeindex[typestr] = idx
        for key, value in self.cellstr_typestr.items():
            cellstr_typeindex[key] = type_list.index(value)
            
        return cellstr_typeindex, typestr_typeindex
    
    def save_gene_cell(self):
        # seperately store gene_name dict and cell_name dict for further use
        # genestr_geneindex : {"C_1": 0}
        # cellstr_cellidnex : {"10001": 1}
        gene_set = {
            n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0
        }
        cell_set = set(self.G) - gene_set
        genestr_geneindex = {
            gene_name: idx for idx, gene_name in enumerate(list(gene_set))
        }
        cellstr_cellindex = {
            cell_name: idx for idx, cell_name in enumerate(list(cell_set))
        }
        return genestr_geneindex, cellstr_cellindex

    def index_convert(self):
        # it is VERY SLOW, but I have no idea of other ways, so I put it here as part of generation of pickle
        nodeindex_geneindex = {}
        geneindex_nodeindex = {}
        for nodestr, nodeindex in self.nodestr_nodeindex.items():
            for genestr, geneindex in self.genestr_geneindex.items():
                    if nodestr == genestr:
                        nodeindex_geneindex[nodeindex] = geneindex
                        geneindex_nodeindex[geneindex] = nodeindex
        return nodeindex_geneindex, geneindex_nodeindex
    
    def get_topktypedict(self, chosen_typestr):
        # build chosen type dict 
        topkclassesstr_topkclassesindex = {}
        for typestr in chosen_typestr:
            topkclassesstr_topkclassesindex[typestr] = self.typestr_typeindex[typestr]
        return topkclassesstr_topkclassesindex
                    
    def get_gene_cell_num(self):
        gene_num = len(self.genestr_geneindex)
        cell_num = len(self.cellstr_cellindex)
        return gene_num, cell_num

    def get_gene_embed(self):
        for n in self.genestr_geneindex:
            gene_embed = torch.zeros(self.gene_num)
            gene_embed[self.genestr_geneindex[n]] = 1
            self.G.nodes[n]['embed'] = gene_embed

    def get_cell_embed(self):
        cell_embed = {}
        for n in self.cellstr_cellindex:
            cell_embed = torch.zeros(self.gene_num)
            for m in self.genestr_geneindex:
                if self.G.get_edge_data(
                        m, n, default=0) and self.G[m][n]['weight'] > 0:
                    cell_embed[self.genestr_geneindex[m]] = 1
            self.G.nodes[n]['embed'] = cell_embed

    def nx_to_dgl(self):
        # VERY VERY IMPORTANT LINE, from_networkx would break the order forr nodes
        dgl_G = nx.relabel_nodes(self.G, self.nodestr_nodeindex)
        dgl_G = dgl.from_networkx(dgl_G,
                                  edge_attrs=['weight'],
                                  node_attrs=['embed'])
        dgl_G = self.make_bidirected(dgl_G)
        dgl_G.edata['weight'] = self.weight_norm(dgl_G)
        return dgl_G
    
    def weight_norm(self, dgl_G):
        graph = dgl_G.local_var()
        graph.edata['weight'] = dgl_G.edata['weight'].reshape(-1,1)
        graph.ndata['in_deg'] = graph.in_degrees(range(graph.number_of_nodes())).float().unsqueeze(-1)
        graph.update_all(dgl.function.copy_edge('weight', 'edge_w'), dgl.function.sum('edge_w', 'total'),
                         lambda nodes: {'norm': nodes.data['total'] / nodes.data['in_deg']})
        graph.apply_edges(lambda edges: {'weight': edges.data['weight'] / edges.dst['norm']})
        return graph.edata['weight']
       
    def make_bidirected(self, dgl_G):
        src_node_tensor = dgl_G.edges()[0]
        tgt_node_tensor = dgl_G.edges()[1]
        weight_tensor = dgl_G.edata['weight']
        dgl_G.add_edges(tgt_node_tensor, src_node_tensor, {'weight': weight_tensor})
        return dgl_G

    def print_embed(self):
        print(self.genestr_geneindex)
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
        self.train_index, self.valid_index, self.test_index = self.split_cell_nodes() 
    
    def extract_feature(self):
        # we need to change 'embed' into csr_matrix since it is too large to successfully pickle
        feature_tensor = csr_matrix(self.G.ndata['embed'])
        # delete embed from the self.G since embed matrix is too big more than (10000 * 10000)
        # thus, after getting feature_tensor, we delete ndata['embed']
        self.G.ndata.pop('embed')
        return feature_tensor

    def extract_label(self):
        label_list = []
        for key, _ in self.nodestr_nodeindex.items():
            if key in self.cellstr_typeindex.keys():
                label_list += [self.cellstr_typeindex[key]]
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


# ============================ After PICKLE =================================
class FlexibleSpeciesDGLDataset(object):
    """Since 1. the pickle file size is limited and feature tensor would be extremely large when in dense form, 
                we need to unfold csr matrix into dense matrix
             2. SpeciesDGLDataset provide us with a complete bipartite graph,
                in practice, we might want to use part of genes cells or use part of its cell labels,
                we need this class to provide a more flexible API for model running
       SpeciesDGLDataset (raw DGL data / raw label&index data) ---> UnFoldedDataset (raw DGL data / modified label&index data) ---> Runner (train/eval)
       
       SpeciesDGLDataset is a general API
       FlexiDatbelSpeciesDGLDataset is a more specially made API
       
    """

    def __init__(self, pkl_file_location, deleted_num=0):
        # preprocessing
        dataset = load_from_pickle(pkl_file_location)
        self.nodestr_nodeindex = dataset.nodestr_nodeindex
        self.genestr_geneindex = dataset.genestr_geneindex
        
        self.nodeindex_geneindex, self.geneindex_nodeindex = dataset.nodeindex_geneindex, dataset.geneindex_nodeindex
        self.topkclassesstr_topkclassesindex = dataset.topkclassesstr_topkclassesindex
        self.classification_report_typestr = self.get_classification_report_typestr()
        
        self.gene_num = dataset.gene_num
        self.cell_num = dataset.cell_num
        self.g, csr_features, self.labels, self.train_index, self.valid_index, self.test_index = dataset.load_data()
        
        # modify feature
        self.features = self.feature_todense(csr_features)
        self.print_info()
    
        # delete required number of genes (randomly delete)
        self.delete_genes(deleted_num)
        self.print_info()
        
    def print_info(self):
        print("================================================================")
        print("[GRAPH] {}".format(self.g))
        print("[FEATURE] {}".format(self.features.shape))
        print("[LABEL] {}".format(self.labels.shape))
        print("[GENE_NUM] {}".format(self.gene_num))
        print("[CELL_NUM] {}".format(self.cell_num))
        print("================================================================")
    
    def feature_todense(self, csr_features):
        return torch.tensor(csr_features.todense())
    
    def get_classification_report_typestr(self):
        # want to change {'a':1, 'b':0, 'c':2} into ['b', 'a', 'c']
        typestr_list = []
        for num in range(len(self.topkclassesstr_topkclassesindex)):
            for key,value in self.topkclassesstr_topkclassesindex.items():
                if value == num:
                    typestr_list.append(key)
        return typestr_list
   
    def delete_nodes(self, g, deleted_nodes):
        g.remove_nodes(deleted_nodes)
        return g
    
    def delete_labels(self, labels, deleted_nodes):
        labels = labels.tolist()
        for i in deleted_nodes.tolist()[::-1]:
            labels.pop(i)
        labels = torch.tensor(labels, dtype=int)
        return labels
    
    def regenerate_features(self, features, deleted_nodes):
        # delete genes from the (gene_num + cell_num) rows 
        # since deleted_nodes are gene indexes in cellstr_cellindex, we can directly delete them
        deleted_nodes_within_genes_cells = deleted_nodes
        remained_rows = [row_id for row_id in range(features.size(0)) if row_id not in deleted_nodes_within_genes_cells]
        features = features[remained_rows]
        
        # delete genes from (gene_num) columns
        deleted_nodes_within_genes = [ self.nodeindex_geneindex[node.item()] for node in deleted_nodes ]
        remained_columns = [col_id for col_id in range(features.size(1)) if col_id not in deleted_nodes_within_genes]
        features = features[:, remained_columns]
        return features
       
    def regenerate_gene_cell_num(self, labels):
        index = 0
        for label in labels:
            if label == -1:
                index += 1
        gene_num = index
        cell_num = len(labels) - gene_num
        return gene_num, cell_num
    
    def resplit_dataset(self, labels):
        cell_idx_list = []
        for idx, label in enumerate(labels):
            if label != -1:
                cell_idx_list.append(idx)
        train_subset, valid_subset, test_subset = dgl.data.utils.split_dataset(torch.tensor(cell_idx_list), shuffle=True, frac_list=[0.7, 0.2, 0.1])
        train_index, valid_index, test_index = train_subset[:], valid_subset[:], test_subset[:]
        return train_index, valid_index, test_index

    def delete_genes(self, deleted_num=0):
        if deleted_num == 0:
            pass
        else:
            # remove randomly sampled nodes from graph
            gene_ids = [nodeindex for nodestr, nodeindex in self.nodestr_nodeindex.items() if nodestr in self.genestr_geneindex.keys()]
            deleted_genes = torch.tensor(sorted(random.sample(gene_ids, deleted_num)))
            
            self.g = self.delete_nodes(self.g, deleted_genes)
            self.labels = self.delete_labels(self.labels, deleted_genes)
            self.features = self.regenerate_features(self.features, deleted_genes)
            self.train_index, self.valid_index, self.test_index = self.resplit_dataset(self.labels)
            self.gene_num, self.cell_num = self.regenerate_gene_cell_num(self.labels)  
            
    def load_data(self):
        return self.g, self.features, self.labels, self.train_index, self.valid_index, self.test_index

