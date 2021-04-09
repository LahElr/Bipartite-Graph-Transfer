# must use in the same directory with model.py
import os
import time
from pathlib import Path
from utils.dataset import FileNxDataset, SpeciesNxDataset, SpeciesDGLDataset
from utils.dataset import save_as_pickle, load_from_pickle

if __name__ == '__main__':
    toy_tuple_list = []
    toy_tuple_list.append(('human','Lung', '1'))
    toy_tuple_list.append(('human','Lung', '2'))
    toy_data_path = Path(os.path.join(os.getcwd(), './data/toy_data'))

    toy_subgraph = []
    for toy_tuple in toy_tuple_list:
        start = time.time()
        toy_subgraph.append(
            FileNxDataset(toy_tuple[0], toy_tuple[1], toy_tuple[2],
                          toy_data_path))
        end = time.time()
        print('[SUBG] save {}th subgraph spent : {}s'.format(
            len(toy_subgraph), end - start))
    
    '''
    for i in range(len(toy_subgraph)):
        gene_num = 0
        cell_num = 0
        print(toy_subgraph[i].G.nodes())
        for n, d in toy_subgraph[i].G.nodes(data=True):
            if d['bipartite'] == 0:
                gene_num += 1
            elif d['bipartite'] == 1:
                cell_num += 1
        print(gene_num, cell_num)
    '''
    
    start = time.time()
    toy_dgl_data = SpeciesDGLDataset([graph for graph in toy_subgraph])
    end = time.time()
    print('[DGLG] save species dgl data : {}s'.format(end - start))

    # NOTICE: when it comes to pikcle, its dump location should be the same with its load location
    start = time.time()
    save_as_pickle(toy_dgl_data, './pkl_data/toy_dgl_data.pkl')
    end = time.time()
    print('[PKLG] save pickle data : {}s'.format(end - start))

    dgl_data = load_from_pickle('./pkl_data/toy_dgl_data.pkl')
