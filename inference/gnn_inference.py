"""

Defines GNNInference objects: models that perform
inference, given a graphical model.
Authors: markcheu@andrew.cmu.edu, kkorovin@cs.cmu.edu

Options:
- Gated Graph neural network:
https://github.com/thaonguyen19/gated-graph-neural-network-pytorch/blob/master/model_pytorch.py
- TBA

"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from inference.core import Inference
from inference.ggnn_model_sparse import GGNN as GGNN_sparse
from inference.ggnn_model import GGNN


class GatedGNNInference(Inference):
    def __init__(self, mode, state_dim, message_dim, 
                hidden_unit_message_dim, hidden_unit_readout_dim, 
                n_steps=10, load_path=None, sparse=False):
        Inference.__init__(self, mode)
        self.model = GGNN(state_dim, message_dim,
                  hidden_unit_message_dim,
                  hidden_unit_readout_dim, n_steps) 
        if sparse:
            self.model = GGNN_sparse(state_dim, message_dim,
                      hidden_unit_message_dim,
                      hidden_unit_readout_dim, n_steps) 

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))
            self.model.eval()
        self.history = {"loss": []}
        self.batch_size = 10

    def run_one(self, graph, device):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode 
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            out,_ = self.model(J,b)
            return out.detach().cpu().numpy()

    def run(self, graphs, device, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = graphs if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph, device))
        return res

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def train(self, dataset, optimizer, criterion, device):
        """ One epoch of training """
        # TODO: set self.batch_size depending on device type
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        batch_loss = []
        mean_losses = []

        for i, graph in enumerate(dataset):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            node_belief, node_feat = self.model(J,b)  # N x 2 with N is the number of nodes

            if self.mode == "marginal":
                target = torch.from_numpy(graph.unary_marginal).float().to(device)
                loss = criterion(node_belief, target)
            else:
                temp = graph.unary_map_onehot 
                temp_2 = np.argmax(temp, axis=1)
                target = torch.from_numpy(temp_2).float().to(device)
                loss = criterion(node_belief.to(torch.float32), target)

            batch_loss.append(loss)

            if (i % self.batch_size == 0):
                ll_mean = torch.stack(batch_loss).mean()
                ll_mean.backward()
                optimizer.step()
                self.model.zero_grad()
                batch_loss=[]
                mean_losses.append(ll_mean.item())
            

        self.history["loss"].append(np.mean(mean_losses))
    
