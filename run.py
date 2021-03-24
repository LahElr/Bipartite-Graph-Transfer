import dgl
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.dataset import load_from_pickle
from model import WeightedGraphSAGE, GraphSAGE, NeighborSampler

# TODO: checkout the relationshihp between fanout and len(input_nodes)

class Runner(object):
    """Used for model training, provide eval(), train() function for object to use

    """

    def __init__(self, pkl_file_location):
        # preprocessing
        dgl_data = load_from_pickle(pkl_file_location)
        g, csr_features, labels, train_index, valid_index, test_index = dgl_data.load_data()

        # modify feature and labels
        class_num = 3
        labels = self.choose_class(labels, class_num)
        features = self.feature_todense(csr_features)

        # data config
        self.g = g
        self.features = features
        self.labels = labels
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.num_workers = 4  # for dataloader init
        self.sampler = self.get_sampler([5, 10], self.g)  # for dataloader init

        # training config
        self.epochs = 600
        self.batch_size = 256
        self.device = 'cpu'

        # model config
        feat_size = self.features.shape[1]
        n_hidden = 1024
        n_classes = torch.max(self.labels).item() + 1
        n_layers = 2
        activation = F.relu
        dropout = 0.5
        self.model = WeightedGraphSAGE(feat_size, n_hidden, n_classes, n_layers,
                               activation, dropout).to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-3)

    def choose_class(self, labels, class_num):
        # help choose the top #class_num kinds of cells to classify
        # change labels into list of 0,1,2,...,class_num-1
        dict = {}
        for key in labels.tolist():
            dict[key] = dict.get(key, 0) + 1  # dict value default 0, add 1 each time

        # choose class_num classes in labels
        chosen_dict_element = sorted(dict.items(), key = lambda x:x[1])[-class_num : ]
        chosen_labels = [dict_tuple[0] for dict_tuple in chosen_dict_element]

        modified_labels = []
        for i in labels.tolist():
            if i in chosen_labels:
                modified_labels.append(chosen_labels.index(i)+1)
            else:
                modified_labels.append(0)
        modified_labels = torch.tensor(modified_labels).reshape(-1)
        return modified_labels

    def feature_todense(self, csr_features):
        return torch.tensor(csr_features.todense())

    def get_sampler(self, fan_out_list, g):
        return NeighborSampler(g, fan_out_list)

    def get_dataloader(self, index, batch_size, sampler, num_workers):
        return DataLoader(dataset=index.numpy(),
                          batch_size=batch_size,
                          collate_fn=sampler.sample_blocks,
                          shuffle=True,
                          drop_last=False,
                          num_workers=num_workers)

    def compute_loss(self, logits, labels):
        return self.loss(logits, labels)

    def compute_acc(self, logits, labels):
        return (torch.argmax(logits, dim=1)
                == labels).float().sum() / len(logits)

    def load_subtensor(self, features, labels, input_nodes, seeds, device):
        # put input_nodes features (subgraph all nodes feature) onto GPU
        # put seeds labels (minibatch nodes label) onto GPU
        batch_inputs = features[input_nodes].to(device)
        batch_labels = labels[seeds].to(device)
        return batch_inputs, batch_labels

    def eval(self):
        acc = []
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
                # Compute loss and prediction
                batch_preds = self.model(blocks, batch_inputs)
                acc.append(self.compute_acc(batch_preds, batch_labels))
        return np.mean(acc)

    def train(self):
        # Training loop
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
                # Compute loss and prediction
                batch_logits = self.model(blocks, batch_inputs)
                loss = self.compute_loss(batch_logits, batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 5 == 0:
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


if __name__ == '__main__':
    hyper_lr_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    hyper_weight_decay_list = [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5,1e-6]
    runner = Runner('./pkl_data/human_dgl_data.pkl')
    runner.train()
