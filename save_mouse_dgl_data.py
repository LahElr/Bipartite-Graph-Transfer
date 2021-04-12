# muse use in the same directory with model.py
import os
import time
from pathlib import Path
from utils.dataset import FileNxDataset, SpeciesNxDataset, SpeciesDGLDataset
from utils.dataset import save_as_pickle, load_from_pickle

if __name__ == '__main__':
    mouse_tuple_list = []
    '''
    mouse_tuple_list.append(('human','Lung', '1'))
    mouse_tuple_list.append(('human','Lung', '2'))
    mouse_data_path = Path(os.path.join(os.getcwd(), './toy_data'))
    '''
    mouse_tuple_list.append(('mouse', 'lung', '1414'))
    mouse_tuple_list.append(('mouse', 'lung', '2512'))
    mouse_tuple_list.append(('mouse', 'lung', '3014'))
    mouse_tuple_list.append(('mouse', 'kidney', '4682'))
    mouse_data_path = Path(os.path.join(os.getcwd(), './data/clean_data/mouse'))

    mouse_subgraph = []
    for mouse_tuple in mouse_tuple_list:
        start = time.time()
        mouse_subgraph.append(
            FileNxDataset(mouse_tuple[0], mouse_tuple[1], mouse_tuple[2],
                          mouse_data_path))
        end = time.time()
        print('[SUBG] save {}th subgraph spent : {}s'.format(
            len(mouse_subgraph), end - start))

    start = time.time()
    mouse_dgl_data = SpeciesDGLDataset([graph for graph in mouse_subgraph])
    end = time.time()
    print('[DGLG] save species dgl data : {}s'.format(end - start))

    # NOTICE: when it comes to pikcle, its dump location should be the same with its load location
    start = time.time()
    save_as_pickle(mouse_dgl_data, './pkl_data/mouse_dgl_data_lung_kidney.pkl')
    end = time.time()
    print('[PKLG] save pickle data : {}s'.format(end - start))

    dgl_data = load_from_pickle('./pkl_data/mouse_dgl_data_lung_kidney.pkl')
