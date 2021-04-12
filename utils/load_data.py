import dgl
import os
import networkx as nx
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class FileBiGraphDataset(object):
    """In tis BiGraph, we think src nodes are genes and tgt nodes are cells

    """
    def __init__(self, species, tissue, number, data_path):
        self.G = nx.Graph()
        celltype_file, data_file = self.get_file_path(species, tissue, number, data_path)
        tgt_list = self.get_tgt(celltype_file, number)
        src_list = self.get_src(data_file)
        weight_mat = self.get_weight(data_file)
        src_tgt_weight_list = self.tuple_construct(src_list, tgt_list, weight_mat)
        self.G.add_weighted_edges_from(src_tgt_weight_list)
        self.G = self.add_attr_celltype(number, celltype_file, self.G)
        print(self.G.nodes())

    def get_file_path(self, species, tissue, number, data_path):
        celltype_file_name = f'{species}_{tissue}{number}_celltype.csv'
        data_file_name = f'{species}_{tissue}{number}_data.csv'
        celltype_file = data_path / celltype_file_name
        data_file = data_path / data_file_name
        return celltype_file, data_file

    def get_tgt(self, cell_type_file, number):
        cell_name_df = pd.read_csv(cell_type_file, usecols=[0])
        cell_name_list = cell_name_df.values[:,0].tolist()
        cell_name_list = [cell_name + '_' + number for cell_name in cell_name_list]
        return cell_name_list

    def get_src(self, data_file):
        gene_name_df = pd.read_csv(data_file, usecols=[0])
        gene_name_list = gene_name_df.values[:,0].tolist()
        return gene_name_list

    def get_weight(self, data_file):
        gene_cell_mat_df = pd.read_csv(data_file)
        gene_cell_mat = gene_cell_mat_df.values[:,1:]
        return gene_cell_mat

    def tuple_construct(self,src_list, tgt_list, weight_mat):
        tuple_list = []
        for i in range(len(src_list)):
            for j in range(len(tgt_list)):
                tuple_list += [(src_list[i], tgt_list[j], weight_mat[i][j])]
                tuple_list += [(tgt_list[j], src_list[i], weight_mat[i][j])]
        return tuple_list

    def add_attr_celltype(self, number, cell_type_file, bigraph):
        cell_df = pd.read_csv(cell_type_file, usecols=[0,1])
        cell_dict = cell_df.set_index('Cell').T.to_dict('records')[0]
        for key, value in cell_dict.items():
            bigraph.nodes[key + '_' + number]['cell_type'] = str(value)
            bigraph.nodes[key + '_' + number]['color'] = 'blue'
        return bigraph

    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.G, ax=ax, with_labels=True)
        plt.show()

class SpeciesBiGraphDataset(object):
    def __init__(self, filebigraph_list):
        graph_list = [graph.G for graph in filebigraph_list]
        self.G = nx.compose_all(graph_list)

    def nx_to_dgl(self):
        return dgl.from_networkx(self.G)

    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.G, ax=ax, with_labels=True)
        plt.show()

if __name__ == '__main__':
    data_path = Path(os.path.join(os.getcwd(), '../data'))
    graph1 = FileBiGraphDataset('human', 'Lung', '1', data_path)
    graph2 = FileBiGraphDataset('human', 'Lung', '2', data_path)
    graph3 = SpeciesBiGraphDataset([graph1, graph2])
    print(graph1.G.nodes['C_1_1'])
    print(graph3.G[10003])
    dgl_G = graph3.nx_to_dgl()
