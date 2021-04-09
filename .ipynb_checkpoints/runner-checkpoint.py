import argparse
import dgl
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from utils.dataset import load_from_pickle, FlexibleSpeciesDGLDataset
from unidomain_model import WeightedGraphSAGE, WeightedGraphSAGEwithEmbed, GraphSAGE, GraphSAGEwithEmbed
from model_utils import NeighborSampler
from sklearn.metrics import classification_report
from dgl.data.utils import load_graphs

# TODO: checkout the relationshihp between fanout and len(input_nodes)


class Runner(object):
    """Used for model training, provide eval(), train() function for object to use
       For single model training instead of transfer training

    """

    def __init__(self, args):
        self._set_seed(args.seed)
        
        # preprocess data
        dataset = FlexibleSpeciesDGLDataset(args.pkl_file_location, args.deleted_gene_num)
        g, features, labels, train_index, valid_index, test_index = dataset.load_data()
    
        # data config
        self.g = g
        self.labels = labels
        #self.features = features
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.num_workers = 4  # for dataloader init
        self.sampler = self.get_sampler([self.g.number_of_nodes()], self.g)  # for dataloader init
        self.classification_report_typestr = dataset.classification_report_typestr # for eval results classification report usage

        # training config
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device 

        # model config
        n_hidden = args.n_hidden
        n_classes = len(self.classification_report_typestr)
        n_layers = args.n_layers
        activation = F.relu
        feat_drop = args.feat_drop
        
        if args.weighted and args.one_hot_feat:
            self.features = features
            feat_size = self.features.shape[1]
            self.model = WeightedGraphSAGEwithEmbed(feat_size, n_hidden, n_classes, n_layers,
                                           activation, feat_drop).to(self.device)
            
        elif args.weighted and not args.one_hot_feat:
            self.features = torch.randn(features.shape[0], 400)
            feat_size = self.features.shape[1]
            self.model = WeightedGraphSAGE(feat_size, n_hidden, n_classes, n_layers,
                                   activation, feat_drop).to(self.device)
            
        elif not args.weighted and args.one_hot_feat:
            self.features = features
            feat_size = self.features.shape[1] 
            self.model = GraphSAGEwithEmbed(feat_size, n_hidden, n_classes, n_layers,
                                           activation, feat_drop).to(self.device)
            
        elif not args.weighted and not args.one_hot_feat:
            self.features = torch.randn(features.shape[0], 400)
            feat_size = self.features.shape[1]
            self.model = GraphSAGE(feat_size, n_hidden, n_classes, n_layers,
                                   activation, feat_drop).to(self.device)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        

    def get_sampler(self, fan_out_list, g):
        return NeighborSampler(g, fan_out_list)

    def get_dataloader(self, index, batch_size, sampler, num_workers):
        # DataLoader must work on CPU instead of GPU
        # we need to put the output of DataLoader onto GPU
        return DataLoader(dataset=index.numpy(),
                          batch_size=batch_size,
                          collate_fn=sampler.sample_blocks,
                          shuffle=True,
                          drop_last=False,
                          num_workers=num_workers)

    def compute_loss(self, logits, labels):
        return self.loss(logits, labels)

    def compute_acc(self, logits, labels):
        return ((torch.argmax(logits, dim=1) == labels).float().sum() /
                len(logits)).detach().cpu()

    def load_subtensor(self, features, labels, input_nodes, seeds, device):
        # put input_nodes features (subgraph all nodes feature) onto GPU
        # put seeds labels (minibatch nodes label) onto GPU
        batch_inputs = features[input_nodes].to(device)
        batch_labels = labels[seeds].to(device)
        return batch_inputs, batch_labels

    def run(self):
        self.train()
        self.test()
        return
    
    def train(self):
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            dataloader = self.get_dataloader(self.train_index, self.batch_size,
                                             self.sampler, self.num_workers)
            for step, blocks in enumerate(dataloader):
                # input_nodes ==> all subgraph train nodes input_nodes[:batch_size] = seeds
                input_nodes = blocks[0].srcdata[dgl.NID]
                # seeds ==> target updating minibatch train nodes
                seeds = blocks[-1].dstdata[dgl.NID]
                # Load the subgraph all nodes features as well as minibatch nodes labels
                batch_inputs, batch_labels = self.load_subtensor(
                    self.features, self.labels, input_nodes, seeds, self.device)
                # Compute loss and prediction, guarantee blocks are on GPU at the same time
                blocks = [block.to(self.device) for block in blocks]
                batch_logits = self.model(blocks, batch_inputs)
                loss = self.compute_loss(batch_logits, batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 50 == 0:
                    acc = self.compute_acc(batch_logits, batch_labels)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated(
                    ) / 1000000 if torch.cuda.is_available() else 0
                    print(
                        'Epoch {:05d} | batch {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MiB'
                        .format(epoch, step, loss.item(), acc.item(),
                                gpu_mem_alloc))

            if epoch != 0:
                eval_acc = self.eval()
                print('Eval Acc {:.4f}'.format(eval_acc))
        return
    
    def eval(self):
        acc = []
        labels = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            dataloader = self.get_dataloader(self.valid_index, self.batch_size,
                                             self.sampler, self.num_workers)
            for step, blocks in enumerate(dataloader):
                # input_nodes ==> all subgraph valid nodes input_nodes[:batch_size] = seeds
                input_nodes = blocks[0].srcdata[dgl.NID]
                # seeds ==> target updating minibatch valid nodes
                seeds = blocks[-1].dstdata[dgl.NID]
                # Load the input features as well as output labels
                batch_inputs, batch_labels = self.load_subtensor(
                    self.features, self.labels, input_nodes, seeds, self.device)
                # Compute loss and prediction, guarantee blocks are on GPU at the same time
                blocks = [block.to(self.device) for block in blocks]
                batch_preds = self.model(blocks, batch_inputs)
                acc.append(self.compute_acc(batch_preds, batch_labels))
                # print validation metric every batch
                labels += batch_labels.tolist()
                preds += torch.argmax(batch_preds, dim=1).tolist()

        print(classification_report(labels, preds, target_names=self.classification_report_typestr))
        return np.mean(acc)
    
    def test(self):
        acc = []
        labels = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            dataloader = self.get_dataloader(self.test_index, self.batch_size,
                                             self.sampler, self.num_workers)
            for step, blocks in enumerate(dataloader):
                # input_nodes ==> all subgraph valid nodes input_nodes[:batch_size] = seeds
                input_nodes = blocks[0].srcdata[dgl.NID]
                # seeds ==> target updating minibatch valid nodes
                seeds = blocks[-1].dstdata[dgl.NID]
                # Load the input features as well as output labels
                batch_inputs, batch_labels = self.load_subtensor(
                    self.features, self.labels, input_nodes, seeds, self.device)
                # Compute loss and prediction, guarantee blocks are on GPU at the same time
                blocks = [block.to(self.device) for block in blocks]
                batch_preds = self.model(blocks, batch_inputs)
                acc.append(self.compute_acc(batch_preds, batch_labels))
                # print validation metric every batch
                labels += batch_labels.tolist()
                preds += torch.argmax(batch_preds, dim=1).tolist()
        print('TEST:')
        print(classification_report(labels, preds, target_names=self.classification_report_typestr))
        return np.mean(acc)


class TransferRunner(object):

    def __init__(self, args):
        # data config
        pkl_file_location = {}
        dataset = {}
        self.g = {}
        self.features = {}
        self.labels = {}
        self.train_index = {}
        self.valid_index = {}
        self.test_index = {}
        self.sampler = {}
        self.num_workers = {}
        self.classification_report_typestr = {}
       
        pkl_file_location['src'] = args.src_pkl_file_location
        pkl_file_location['tgt'] = args.tgt_pkl_file_location
       
        for domain in ['src', 'tgt']:
            dataset[domain] = FlexibleSpeciesDGLDataset(pkl_file_location[domain])
            self.g[domain], self.features[domain], self.labels[domain], self.train_index[domain], self.valid_index[domain], self.test_index[domain] = dataset[domain].load_data()
            self.num_workers[domain] = 4  # for dataloader init
            self.sampler[domain] = self.get_sampler([self.g[domain].number_of_nodes()], self.g[domain])  # for dataloader init
            self.classification_report_typestr[domain] = dataset[domain].classification_report_typestr # f
        
        self.features = self.pick_domain_shared_genes_in_onehot_feature_columns(dataset)
        
        # training config
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device 
      
        # model config
        n_hidden = args.n_hidden
        n_classes = torch.max(self.labels).item() + 1
        n_layers = args.n_layers
        activation = F.relu
        feat_drop = args.feat_drop
        
        self.model = TransferGraphSAGE(feat_size, n_hidden, n_classes, n_layers,
                                       activation, dropout).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-4)
        
    def pick_domain_shared_genes_in_onehot_feature_columns(self, dataset):
        # in the feature shaped by (gene_num + cell_num, gene_num) embedding
        # we only choose shared_genes in gene_num columns
        # in the ranking of nodes in feature:
        #    1.rows are listed based on nodestr_nodeindex
        #    2.cols are listed based on genestr_geneindex
        shared_gene_idx_within_genes = self.find_shared_gene_within_genes(dataset)
        self.features = self.modify_feature_dim(shared_gene_idx_within_genes, self.features)
        assert self.features['src'].shape[1] == self.features['tgt'].shape[1]
        return self.features
        
        
    def find_shared_gene_within_genes(self, dataset):
        src_nodes_set = set(dataset['src'].genestr_geneindex.keys())
        tgt_nodes_set = set(dataset['tgt'].genestr_geneindex.keys())
        shared_genestr_list = list(src_nodes_set & tgt_nodes_set)
        shared_geneindex = {}
        for domain in ['src', 'tgt']:
            shared_geneindex[domain] = [dataset[domain].genestr_geneindex[genestr] for genestr in shared_genestr_list]
        return shared_geneindex
    
    def find_shared_gene_within_genes_cells(self, dataset):
        src_nodes_set = set(dataset['src'].nodestr_nodeindex.keys())
        tgt_nodes_set = set(dataset['tgt'].nodestr_nodeindex.keys())
        shared_genestr_list = list(src_nodes_set & tgt_nodes_set)
        shared_geneindex = {}
        for domain in ['src', 'tgt']:
            shared_geneindex[domain] = [dataset[domain].nodestr_nodeindex[genestr] for genestr in shared_genestr_list]
        return shared_geneindex
        

    def modify_feature_dim(self, shared_gene_idx, features):
        for domain in ['src', 'tgt']:
            features[domain] = features[domain][:, shared_gene_idx[domain]]
        return features
    
    def get_sampler(self, fan_out_list, g):
        return NeighborSampler(g, fan_out_list)

    def get_dataloader(self, index, batch_size, sampler, num_workers):
        # DataLoader must work on CPU instead of GPU
        # we need to put the output [block in list of blocks] of DataLoader onto GPU
        return DataLoader(dataset=index.numpy(),
                          batch_size=batch_size,
                          collate_fn=sampler.sample_blocks,
                          shuffle=True,
                          drop_last=False,
                          num_workers=num_workers)

    def eval(self):
        return

    def train(self):
        return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--pkl_file_location", type=str, default='./pkl_data/human_dgl_data_kidney.pkl')
    
    parser.add_argument("--feat_drop", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_hidden", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=1)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda')
    
    parser.add_argument("--weighted", action='store_true', default=False)
    parser.add_argument("--one_hot_feat", action='store_true', default=False)
    
    parser.add_argument("--deleted_gene_num", type=int, default=0)
    
    args = parser.parse_args()
    runner = Runner(args)
    runner.run()
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--src_pkl_file_location", type=str, default='./pkl_data/human_dgl_data_lung.pkl')
    parser.add_argument("--tgt_pkl_file_location", type=str, default='./pkl_data/mouse_dgl_data_lung.pkl')
    
    parser.add_argument("--feat_drop", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--n_hidden", type=int, default=200)
    parser.add_argument("--n_layers", type=int, default=1)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default='cuda')
    
    args = parser.parse_args()
    transfer_runner = TransferRunner(args)
    '''
    
    
