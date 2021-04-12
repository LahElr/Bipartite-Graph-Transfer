import argparse

import random
import time
import pickle
import bidict
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from dgl.data.utils import load_graphs

from utils.dataset import load_from_pickle, FlexibleSpeciesDGLDataset
from unidomain_model import WeightedGraphSAGE, WeightedGraphSAGEwithEmbed, GraphSAGE, GraphSAGEwithEmbed
from unidomain_model import GraphTransformerNet, WeightedGraphTransformerNet
from model_utils import NeighborSampler
from utils import lahelr_print

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# // TODO: checkout the relationshihp between fanout and len(input_nodes)


class Runner(object):
    """Used for model training, provide eval(), train() function for object to use
       For single model training instead of transfer training

    """

    def __init__(self, args):
        self._set_seed(args.seed)
        self.device = args.device

        # preprocess data
        dataset = FlexibleSpeciesDGLDataset(args.pkl_file_location)
        g, features, labels, train_index, valid_index, test_index = dataset.load_data()

        # =========================== test to delete certain number of genes ===========================
        print("the graph before deleting gnens:", g, sep="\n")
        gene_idx_list = []
        for idx, label in enumerate(labels):
            if label == -1:
                gene_idx_list.append(idx)

        if args.deleted_gene_num > 0:
            print("Delete", args.deleted_gene_num, "genes.")
            deleted_gene_list = torch.tensor(
                sorted(random.sample(gene_idx_list, args.deleted_gene_num)))
            print("The list of deleted gene:", deleted_gene_list)
            g.remove_nodes(deleted_gene_list)
            print("The graph after removing nodes:", g, sep="\n")

            print(
                "The feature of the graph before removing genes is of shape:", features.shape)
            features = features.tolist()
            for i in deleted_gene_list.tolist()[::-1]:
                features.pop(i)
            features = torch.FloatTensor(features)
            print(
                "The feature of the graph after removing genes is of shape:", features.shape)

            print(
                "The labels of the nodes before removing gnens is of shape:", labels.shape)
            labels = labels.tolist()
            for i in deleted_gene_list.tolist()[::-1]:
                labels.pop(i)
            labels = torch.tensor(labels, dtype=int)
            print(
                "The labels of the nodes after removing gnens is of shape:", labels.shape)

            index = 0
            for label in labels:
                if label == -1:
                    index += 1
            self.gene_num = index
            print("There are", self.gene_num, "genes in the graph now.")
            print("There are", len(labels) -
                  self.gene_num, "cells in the graph now.")

            cell_idx_list = []
            for idx, label in enumerate(labels):
                if label != -1:
                    cell_idx_list.append(idx)
            train_subset, valid_subset, test_subset = dgl.data.utils.split_dataset(
                torch.tensor(cell_idx_list), shuffle=True, frac_list=[0.7, 0.2, 0.1])
            train_index = train_subset[:]
            valid_index = valid_subset[:]
            test_index = test_subset[:]
        else:
            print("No deleting nodes")

        # ==fin==fin==fin==fin==fin== test to delete certain number of genes ==fin==fin==fin==fin==fin==

        # data config
        self.g = g
        if args.one_hot_feat:
            self.features = features
        else:
            self.features = torch.randn(features.shape)
        self.g.ndata["feat"] = self.features  # .int()
        # .repeat(1, *(list(self.features.shape)[1:]))
        self.g.edata["weight"] = self.g.edata["weight"]
        self.labels = (labels + 1).to(self.device)
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.num_workers = 0  # for dataloader init
        self.sampler = self.get_sampler([5, 5], self.g)  # for dataloader init

        print('gene_num : {}'.format(dataset.gene_num))
        print('cell_num : {}'.format(dataset.cell_num))

        # training config
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = args.device
        self.use_scheduler = args.use_scheduler

        # model config
        net_params = {
            "device": self.device,
            "batch_size": self.batch_size,
            "L": args.n_layers,
            "n_heads": 2,
            "hidden_dim": args.n_hidden,
            "out_dim": args.out_dim,
            "residual": True,
            "readout": "None",
            "in_feat_dropout": args.in_feat_drop,
            "dropout": args.feat_drop,
            "layer_norm": True,
            "batch_norm": False,
            "self_loop": False,
            "lap_pos_enc": False,
            "wl_pos_enc": False,
            "full_graph": False,
            "edge_feat": args.weighted,
            "pos_enc_dim": 0,
            "max_wl_role_index": 12,
            'weight_pre': 'linear',
            'feat_pre': 'linear'
        }
        # net_params["in_dim_node"] = int(torch.unique(self.g.ndata['feat'], dim=0).size(0))
        # net_params['in_dim_edge'] = int(torch.unique(self.g.edata['weight'], dim=0).size(0))
        net_params["feat_size"] = list(self.g.ndata['feat'][0].size())
        net_params["weight_size"] = list(self.g.edata['weight'][0].size())
        net_params['in_dim_edge'] = net_params["weight_size"][-1]
        net_params['in_dim_node'] = net_params["feat_size"][-1]
        net_params["n_classes"] = torch.unique(self.labels, dim=0).size(
            0)  # torch.max(self.labels).item() + 1

        print("-"*17)
        print("net_params:")
        print(*(["\"{}\" : {}".format(x[0], x[1])
              for x in net_params.items()]), sep="\n")
        print("-"*18)
        print("args:")
        for k in list(vars(args).keys()):
            print('{}: {}'.format(k, vars(args)[k]))
        print("-"*19)

        self.best_loss = 100
        self.best_loss_posi = []
        self.best_acc1 = 0
        self.best_acc1_posi = []
        self.best_acc2 = 0
        self.best_acc2_posi = []
        self.best_eval_acc1 = 0
        self.best_eval_acc1_posi = []
        self.best_eval_acc2 = 0
        self.best_eval_acc2_posi = []

        if args.weighted_transformer:
            self.model = WeightedGraphTransformerNet(
                net_params).to(self.device)
        else:
            self.model = GraphTransformerNet(net_params).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=args.sch_factor, patience=args.sch_patience, verbose=True)

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

    def step_summary(self, epoch, step, loss, acc1, acc2):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_loss_posi = [(epoch, step)]
            print("new best loss {}.".format(loss))
        elif loss == self.best_loss:
            self.best_loss_posi.append((epoch, step))
        if acc1 > self.best_acc1:
            self.best_acc1 = acc1
            self.best_acc1_posi = [(epoch, step)]
            print("new best acc1 {}.".format(acc1))
        elif acc1 == self.best_acc1:
            self.best_acc1_posi.append((epoch, step))
        if acc2 > self.best_acc2:
            self.best_acc2 = acc2
            self.best_acc2_posi = [(epoch, step)]
            print("new best acc2 {}.".format(acc2))
        elif acc2 == self.best_acc2:
            self.best_acc2_posi.append((epoch, step))

    def eval_summary(self, epoch, acc1, acc2):
        if acc1 > self.best_eval_acc1:
            self.best_eval_acc1 = acc1
            self.best_eval_acc1_posi = [epoch]
            print("new best eval acc1 {}.".format(acc1))
        elif acc1 == self.best_eval_acc1:
            self.best_eval_acc1_posi.append(epoch)
        if acc2 > self.best_eval_acc2:
            self.best_eval_acc2 = acc2
            self.best_eval_acc2_posi = [epoch]
            print("new best eval acc2 {}.".format(acc2))
        elif acc2 == self.best_eval_acc2:
            self.best_eval_acc2_posi.append(epoch)

    def compute_acc(self, logits, labels):
        ori_labels = labels
        S = labels.cpu().numpy()
        C = np.argmax(torch.nn.Softmax(dim=1)(logits).cpu().detach().numpy(),
                      axis=1)
        CM = confusion_matrix(S, C).astype(np.float32)
        nb_classes = CM.shape[0]
        labels = labels.cpu().detach().numpy()
        nb_non_empty_classes = 0
        pr_classes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(labels == r)[0]
            if cluster.shape[0] != 0:
                pr_classes[r] = CM[r, r] / float(cluster.shape[0])
                if CM[r, r] > 0:
                    nb_non_empty_classes += 1
            else:
                pr_classes[r] = 0.0
        acc1 = (100. * np.sum(pr_classes) / float(nb_classes))
        acc2 = ((torch.argmax(logits, dim=1) == ori_labels).float().sum() /
                len(logits)).detach().cpu()
        return (acc1, acc2)

    def train(self):
        # Training loop
        scheduler_break = False
        for epoch in range(self.epochs):
            # one epoch start
            self.model.train()
            losses = []

            dataloader = self.get_dataloader(self.train_index, self.batch_size,
                                             self.sampler, self.num_workers)
            max_step = len(dataloader)
            for step, _ in enumerate(dataloader):
                batch_graph = _[0].to(self.device)
                node_map = _[1]
                edge_map = _[2]
                batch_labels = self.labels[list(node_map.values())].to(
                    self.device)
                batch_x = batch_graph.ndata['feat'].to(
                    self.device)  # num x feat
                batch_e = batch_graph.edata['weight'].to(torch.float32).to(
                    self.device)
                self.optimizer.zero_grad()

                try:
                    batch_lap_pos_enc = batch_graph.ndata['lap_pos_enc'].to(
                        self.device)
                    sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(
                        self.device)
                    sign_flip[sign_flip >= 0.5] = 1.0
                    sign_flip[sign_flip < 0.5] = -1.0
                    batch_lap_pos_enc = batch_lap_pos_enc * \
                        sign_flip.unsqueeze(0)
                except:
                    batch_lap_pos_enc = None

                try:
                    batch_wl_pos_enc = batch_graph.ndata['wl_pos_enc'].to(
                        self.device)
                except:
                    batch_wl_pos_enc = None

                # Compute loss and prediction
                batch_scores = self.model.forward(batch_graph, batch_x,
                                                  batch_e, batch_lap_pos_enc,
                                                  batch_wl_pos_enc)
                loss = self.model.loss(batch_scores, batch_labels)
                losses.append(float(loss.clone().detach()))

                loss.backward()
                self.optimizer.step()
                # ? TODO where to put the scheduler is still not decided
                if self.use_scheduler:
                    self.scheduler.step(loss)
                    if self.optimizer.param_groups[0]['lr'] < 1e-6:
                        print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                        scheduler_break = True
                        break

                if step % 5 == 0 or step == max_step:
                    acc = self.compute_acc(batch_scores, batch_labels)
                    acc1 = acc[0]
                    acc2 = acc[1]
                    gpu_mem_alloc = torch.cuda.max_memory_allocated(
                    ) / 1000000 if torch.cuda.is_available() else 0
                    print(
                        'Epoch {:05d} | batch {:05d} | Loss {:.4f} | Train Acc {:.4f} or {:.4f} | GPU {:.1f} MiB'
                        .format(epoch, step, loss.item(), acc2.item(), acc1,
                                gpu_mem_alloc))
                    self.step_summary(
                        epoch, step, loss.item(), acc1, acc2.item())

            # // ? TODO where to put the scheduler is still not decided
            # self.scheduler.step(sum(losses)/len(losses))
            losses.clear()
            if scheduler_break:
                break

            # if epoch != 0:
            eval_acc = self.eval()
            print('Eval Acc {:.4f} or {:.4f}'.format(eval_acc[0], eval_acc[1]))
            self.eval_summary(epoch, eval_acc[0], eval_acc[1])

        print("statistic data:")
        print("best train loss {} at {}.".format(
            self.best_loss, self.best_loss_posi))
        print("best train acc1 {} at {}.".format(
            self.best_acc1, self.best_acc1_posi))
        print("best train acc2 {} at {}.".format(
            self.best_acc2, self.best_acc2_posi))
        print("best eval acc1 {} at {}.".format(
            self.best_eval_acc1, self.best_eval_acc1_posi))
        print("best eval acc2 {} at {}.".format(
            self.best_eval_acc2, self.best_eval_acc2_posi))

        test_acc = self.test()
        print('Test Acc {:.4f} or {:.4f}'.format(test_acc[0], test_acc[1]))

        return

    def eval(self):
        acc2s = []
        acc1s = []
        labels = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            dataloader = self.get_dataloader(self.valid_index, self.batch_size,
                                             self.sampler, self.num_workers)
            for step, _ in enumerate(dataloader):
                batch_graph = _[0].to(self.device)
                node_map = _[1]
                edge_map = _[2]
                batch_labels = self.labels[list(node_map.values())].to(
                    self.device)
                batch_x = batch_graph.ndata['feat'].to(
                    self.device)  # num x feat
                batch_e = batch_graph.edata['weight'].to(torch.float32).to(
                    self.device)
                try:
                    batch_lap_pos_enc = batch_graph.ndata['lap_pos_enc'].to(
                        self.device)
                except:
                    batch_lap_pos_enc = None
                try:
                    batch_wl_pos_enc = batch_graph.ndata['wl_pos_enc'].to(
                        self.device)
                except:
                    batch_wl_pos_enc = None

                batch_preds = self.model.forward(batch_graph, batch_x, batch_e,
                                                 batch_lap_pos_enc,
                                                 batch_wl_pos_enc)
                # loss = self.model.loss(batch_preds,batch_labels)
                acc = self.compute_acc(batch_preds, batch_labels)
                acc1s.append(acc[0])
                acc2s.append(acc[1])
                # print validation metric every batch
                labels += batch_labels.tolist()
                preds += torch.argmax(batch_preds, dim=1).tolist()

        print(classification_report(labels, preds))
        return (torch.mean(torch.Tensor(acc1s)),
                torch.mean(torch.Tensor(acc2s)))

    def test(self):
        acc1s = []
        acc2s = []
        labels = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            dataloader = self.get_dataloader(self.test_index, self.batch_size,
                                             self.sampler, self.num_workers)
            for step, _ in enumerate(dataloader):
                # Compute loss and prediction
                batch_graph = _[0].to(self.device)
                node_map = _[1]
                edge_map = _[2]
                batch_labels = self.labels[list(node_map.values())].to(
                    self.device)
                batch_x = batch_graph.ndata['feat'].to(
                    self.device)  # num x feat
                batch_e = batch_graph.edata['weight'].to(torch.float32).to(
                    self.device)
                try:
                    batch_lap_pos_enc = batch_graph.ndata['lap_pos_enc'].to(
                        self.device)
                except:
                    batch_lap_pos_enc = None
                try:
                    batch_wl_pos_enc = batch_graph.ndata['wl_pos_enc'].to(
                        self.device)
                except:
                    batch_wl_pos_enc = None

                batch_preds = self.model.forward(batch_graph, batch_x, batch_e,
                                                 batch_lap_pos_enc,
                                                 batch_wl_pos_enc)
                # loss = self.model.loss(batch_preds,batch_labels)
                acc = self.compute_acc(batch_preds, batch_labels)
                acc1s.append(acc[0])
                acc2s.append(acc[1])
                # print validation metric every batch
                labels += batch_labels.tolist()
                preds += torch.argmax(batch_preds, dim=1).tolist()
        print('TEST:')
        print(classification_report(labels, preds))
        return (torch.mean(torch.Tensor(acc1s)),
                torch.mean(torch.Tensor(acc2s)))


class TransferRunner(object):

    def __init__(self, src_pkl_file_location, tgt_pkl_file_location):
        # data config
        pkl_file_location, dataset = {}
        self.g, self.features, self.labels = {}, {}, {}
        self.train_index, self.valid_index, self.test_index = {}, {}, {}
        self.sampler, self.num_workers = {}, {}

        pkl_file_location['src'] = src_pkl_file_location
        pkl_file_location['tgt'] = tgt_pkl_file_location

        for domain in ['src', 'tgt']:
            dataset[domain] = FlexibleSpeciesDGLDataset(
                pkl_file_location[domain])
            self.g[domain], self.features[domain], self.labels[domain], self.train_index[
                domain], self.valid_index[domain], self.test_index[domain] = dataset[domain].load_data()
            self.num_workers[domain] = 4  # for dataloader init
            self.sampler[domain] = self.get_sampler(
                [5, 10], self.g[domain])  # for dataloader init

        shared_gene_src_index, shared_gene_tgt_index = self.find_shared_gene(
            dataset['src'], tgt_dataset['tgt'])

        # training config
        self.epochs = 600
        self.batch_size = 256
        self.device = 'cuda'

        # model config
        if self.features['src'].shape[1] != self.features['tgt'].shape[1]:
            raise ValueError('source and target feature shape not correct !')
        feat_size = self.features['src'].shape[1]
        n_hidden = 1024
        n_classes = num_class
        n_layers = 2
        activation = F.relu
        dropout = 0.5
        self.model = WeightedGraphSAGE(feat_size, n_hidden, n_classes, n_layers,
                                       activation, dropout).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-4)

    def find_shared_gene(self, src_dataset, tgt_dataset):
        src_gene_num = src_dataset.gene_num
        tgt_gene_num = tgt_dataset.gene_num
        print(src_dataset.nodestr_nodeindex[:src_gene_num])
        print(tgt_dataset.nodestr_nodeindex[:tgt_gene_num])

        shared_gene_index = []
        return shared_gene_src_index, shared_gene_tgt_index

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
    parser.add_argument("--use_scheduler", action='store_true', default=False)
    parser.add_argument("--pkl_file_location", type=str,
                        default='./pkl_data/mouse_dgl_data_kidney.pkl')

    parser.add_argument("--feat_drop", type=float, default=0.0)
    parser.add_argument("--in_feat_drop", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_hidden", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument("--weighted", action='store_true', default=True)
    parser.add_argument("--one_hot_feat", action='store_true', default=True)
    parser.add_argument("--weighted_transformer",
                        action='store_true', default=True)

    parser.add_argument("--deleted_gene_num", type=int, default=1)
    parser.add_argument("--sch_factor", type=int, default=0.5)
    parser.add_argument("--sch_patience", type=int, default=10)

    args = parser.parse_args()
    #transfer_runner = TransferRunner('./pkl_data/human_dgl_data.pkl', './pkl_data/mouse_dgl_data.pkl')
    runner = Runner(args)
    runner.train()
