# must use in the same directory with model.py
import os
import time
from pathlib import Path
from utils.dataset import FileNxDataset, SpeciesNxDataset, SpeciesDGLDataset
from utils.dataset import save_as_pickle, load_from_pickle

if __name__ == '__main__':
    human_tuple_list = []
    '''
    human_tuple_list.append(('human','Lung', '1'))
    human_tuple_list.append(('human','Lung', '2'))
    human_data_path = Path(os.path.join(os.getcwd(), '../toy_data'))
    '''
    
    human_tuple_list.append(('human', 'lung', '8426'))
    human_tuple_list.append(('human', 'lung', '6022'))
    human_tuple_list.append(('human', 'lung', '9603'))
    human_tuple_list.append(('human', 'kidney', '3849'))
    human_tuple_list.append(('human', 'kidney', '9153'))
    human_tuple_list.append(('human', 'kidney', '9966'))
    human_data_path = Path(os.path.join(os.getcwd(), './data/clean_data/human'))

    human_subgraph = []
    for human_tuple in human_tuple_list:
        start = time.time()
        human_subgraph.append(
            FileNxDataset(human_tuple[0], human_tuple[1], human_tuple[2],
                          human_data_path))
        end = time.time()
        print('[SUBG] save {}th subgraph spent : {}s'.format(
            len(human_subgraph), end - start))
    
    for i in range(len(human_subgraph)):
        gene_num = 0
        cell_num = 0
        for n, d in human_subgraph[i].G.nodes(data=True):
            if d['bipartite'] == 0:
                gene_num += 1
            elif d['bipartite'] == 1:
                cell_num += 1
        print(gene_num, cell_num)
    

    start = time.time()
    human_dgl_data = SpeciesDGLDataset([graph for graph in human_subgraph])
    end = time.time()
    print('[DGLG] save species dgl data : {}s'.format(end - start))

    # NOTICE: when it comes to pikcle, its dump location should be the same with its load location
    start = time.time()
    save_as_pickle(human_dgl_data, './pkl_data/human_dgl_data_lung_kidney.pkl')
    end = time.time()
    print('[PKLG] save pickle data : {}s'.format(end - start))

    dgl_data = load_from_pickle('./pkl_data/human_dgl_data_lung_kidney.pkl')
