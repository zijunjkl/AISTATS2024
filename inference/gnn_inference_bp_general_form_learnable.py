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
import random
from inference.core import Inference

from inference.bpgnn_model_map_general_form_learnable import BPGNN as BPGNN_general_form_learnable
from inference.bpgnn_model_marginal_general_form_learnable import BPGNN as BPGNN_general_form_learnable_marginal
import itertools

def compute_potential_2(state, adj, unary, H):
    potential = 1
    N = len(unary)
    row, col = np.nonzero(adj)
    for i in np.arange(N):
        potential *= unary[i,state[i]]
    for e in np.arange(len(row)):
        ii = row[e]
        jj = col[e]
        eta_ij = H[ii,jj]
        H_ij = np.exp([[eta_ij, -eta_ij],[-eta_ij, eta_ij]])
        potential *= H_ij[state[ii], state[jj]]
    
    return potential

def compute_probs(adj, unary, k, n, H):
    potentials = np.zeros([k]*n)
    for state in itertools.product(np.arange(k), repeat=n):
        state_ind = np.array(state)
        potentials[state] = compute_potential_2(state, adj, unary, H)
    partition = np.sum(potentials)
    probs = potentials/partition
    return probs, np.log(partition)


def compute_mi(p_i, p_j, p_ij):
    mi_ = 0
    for k in range(2):
        for l in range(2):
            mi_ = mi_ + p_ij[k,l]*np.log(p_ij[k,l]/(p_i[k]*p_j[l]))
            
    return mi_

def compute_weighted_matrix(belief_single, belief_pair, adj):
    rows, cols = np.where(adj)
    n_E = len(rows)
    mutual_info_matrix = adj.copy()
    
    for ee in range(n_E):
        i = rows[ee]
        j = cols[ee]
        p_i = belief_single[i]
        p_j = belief_single[j]
        p_ij = belief_pair[ee]
        mutual_info_matrix[i,j] = compute_mi(p_i, p_j, p_ij)
        
    return mutual_info_matrix
    
class MPNNInference_General_GT(Inference):
    def __init__(self, mode, method, state_dim, message_dim, 
                hidden_unit_message_dim, hidden_unit_readout_dim, 
                n_steps=10, load_path=None, sparse=False):
        Inference.__init__(self, mode)

        self.method = method

        if method == 'MPNNgeneral_gt':
            if mode == 'map':
                self.model = BPGNN_general_form_learnable(state_dim, message_dim, 
                        hidden_unit_message_dim,
                        hidden_unit_readout_dim, n_steps)
            elif mode == 'marginal':
                self.model = BPGNN_general_form_learnable_marginal(state_dim, message_dim, 
                        hidden_unit_message_dim,
                        hidden_unit_readout_dim, n_steps)

        if load_path is not None:
            self.model.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage,
                    loc: storage))
            self.model.eval()
            
        self.history = {"loss": [], "entropy":[], "cMSE":[], "ce":[], "reg":[], "message_reg":[]}
        self.batch_size = 50
        

    def run_one(self, graph, N_step, MPNNtype, device):
        """ Forward computation that depends on the mode """
        # Call to super forward
        # wrap up depending on mode 
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            if MPNNtype == 'node':
                out_unary, _, message_arr, entropy_,_,_, Ci, Cij,_ = self.model(J,b, N_step, MPNNtype)
                return out_unary.detach().cpu().numpy(), message_arr, entropy_.detach().cpu().numpy(),\
                    Ci.detach().cpu().numpy(), Cij.detach().cpu().numpy()
            
            elif MPNNtype == 'edge':
                _, out_pairwise, message_arr, entropy_,_,_, Ci, Cij,_ = self.model(J,b, N_step, MPNNtype)
                return out_pairwise.detach().cpu().numpy(), message_arr, entropy_.detach().cpu().numpy(), \
                    Ci.detach().cpu().numpy(), Cij.detach().cpu().numpy()
            
            elif MPNNtype == 'both':
                out_unary, out_pairwise, message_arr, entropy_,_,_, Ci, Cij,_ = self.model(J,b, N_step, MPNNtype)
                return out_unary.detach().cpu().numpy(), out_pairwise.detach().cpu().numpy(), message_arr, entropy_.detach().cpu().numpy(),\
                    Ci.detach().cpu().numpy(), Cij.detach().cpu().numpy()
            
        
    def run(self, graphs, MPNNtype, N_step, device, verbose=False):
        self.verbose = verbose
        res_unary = []
        res_pairwise = []
        res_entropy = []
        res_Ci = []
        res_Cij = []
        res_time = []
        graph_iterator = graphs if self.verbose else graphs
        
        if MPNNtype == 'node':
            for graph in graph_iterator:
                unary_, pairwise_, _, entropy_, Ci, Cij = self.run_one(graph, N_step, MPNNtype, device)
                res_unary.append(unary_)
                res_entropy.append(entropy_)
                res_Ci.append(Ci)
                res_Cij.append(Cij)
            
            return res_unary, res_entropy, res_Ci, res_Cij
            
        elif MPNNtype == 'edge':
            for graph in graph_iterator:
                unary_, pairwise_, _, entropy_ , Ci, Cij = self.run_one(graph, N_step, MPNNtype, device)
                res_pairwise.append(pairwise_)
                res_entropy.append(entropy_)
                res_Ci.append(Ci)
                res_Cij.append(Cij)
            
            return res_pairwise, res_entropy, res_Ci, res_Cij
            
        elif MPNNtype == 'both':
            for graph in graph_iterator:
                unary_, pairwise_, computing_time, entropy_, Ci, Cij = self.run_one(graph, N_step, MPNNtype, device)
                res_unary.append(unary_)
                res_pairwise.append(pairwise_)
                res_entropy.append(entropy_)
                res_Ci.append(Ci)
                res_Cij.append(Cij)
                res_time.append(computing_time)
            
            return res_unary, res_pairwise, res_entropy, res_Ci, res_Cij, res_time
        
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
    
    
    def train(self, dataset, optimizer, criterion, Ci_BP, Cij_BP, N_step, device):
        """ One epoch of training """
        # TODO: set self.batch_size depending on device type
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        batch_loss = []
        mean_losses = []
        entropy_batch_loss = []
        entropy_mean_losses = []
        c_batch_loss = []
        c_mean_losses = []
        
        for i, graph in enumerate(dataset):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            node_belief, _, message_arr,entropy_,_,_,Ci, Cij,dom_scaler = self.model(J,b,N_step, 'node')  # N x 2 with N is the number of nodes
            
            
            # print(node_belief)
            if self.mode == "marginal":
                target = torch.from_numpy(graph.unary_marginal).float().to(device)
                loss = criterion(node_belief, target)
            else:
                temp = graph.unary_map_onehot 
                temp_2 = np.argmax(temp, axis=1)
                target = torch.from_numpy(temp_2).float().to(device)
                loss = criterion(node_belief, target)
                
            Ci_BP_tensor = torch.from_numpy(Ci_BP[i]).float().to(device)
            Cij_BP_tensor = torch.from_numpy(Cij_BP[i]).float().to(device)
        
            c_MSE = torch.norm(Ci-Ci_BP_tensor) + torch.norm(Cij-Cij_BP_tensor)
            batch_loss.append(loss)
            entropy_batch_loss.append(entropy_)
            c_batch_loss.append(c_MSE)
            
            if (i % self.batch_size == 0):
                ll_mean = torch.stack(batch_loss).mean()
                ll_mean.backward()
                optimizer.step()
                self.model.zero_grad()
                batch_loss=[]
                mean_losses.append(ll_mean.item())
            
                entropy_mean = torch.stack(entropy_batch_loss).mean()
                entropy_batch_loss = []
                entropy_mean_losses.append(entropy_mean.item())
                
                c_mean = torch.stack(c_batch_loss).mean()
                c_batch_loss = []
                c_mean_losses.append(c_mean.item())
                
        self.history["loss"].append(np.mean(mean_losses))
        self.history["cMSE"].append(np.mean(c_mean_losses))
        self.history["entropy"].append(np.mean(entropy_mean_losses))



    def train_free_energy(self, dataset, optimizer, Ci_BP, Cij_BP, N_step, device):
        """ One epoch of training """
        # TODO: set self.batch_size depending on device type
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        batch_loss = []
        mean_losses = []
        c_batch_loss = []
        c_mean_losses = []
        Ci_BP_tensor = torch.from_numpy(Ci_BP).float().to(device)
        Cij_BP_tensor = torch.from_numpy(Cij_BP).float().to(device)
        

        for i, graph in enumerate(dataset):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            node_belief, _, _, entropy_,_,_,Ci, Cij, dom_scaler = self.model(J,b, N_step, 'node')  # N x 2 with N is the number of nodes
            
            c_MSE = torch.norm(Ci-Ci_BP_tensor) + torch.norm(Cij-Cij_BP_tensor)
            # print('entropy = %f, c_MSE = %f'%(entropy_, c_MSE))
            batch_loss.append(entropy_ + 0.1*c_MSE)
            c_batch_loss.append(c_MSE)
            
            if (i % self.batch_size == 0):
                ll_mean = torch.stack(batch_loss).mean()
                ll_mean.backward()
                optimizer.step()
                self.model.zero_grad()
                batch_loss=[]
                mean_losses.append(ll_mean.item())
                
                c_mean = torch.stack(c_batch_loss).mean()
                c_batch_loss = []
                c_mean_losses.append(c_mean.item())
                

        self.history["loss"].append(np.mean(mean_losses))
        self.history["cMSE"].append(np.mean(c_mean_losses))
        
        
    def train_pretrain_BP(self, dataset, optimizer, Ci_BP, Cij_BP, device):
        """ One epoch of training """
        # TODO: set self.batch_size depending on device type
        self.model.to(device)
        self.model.train()
        self.model.zero_grad()

        batch_loss = []
        mean_losses = []
        entropy_batch_loss = []
        entropy_mean_losses = []
        
            
        for i, graph in enumerate(dataset):
            b = torch.from_numpy(graph.b).float().to(device)
            J = torch.from_numpy(graph.W).float().to(device)
            
            Ci_BP_tensor = torch.from_numpy(Ci_BP[i]).float().to(device)
            Cij_BP_tensor = torch.from_numpy(Cij_BP[i]).float().to(device)
        
            node_belief, edge_belief, _, entropy_,entropy_node,entropy_edge,Ci, Cij, dom_scaler = self.model(J,b, 'node')  # N x 2 with N is the number of nodes
            
            # ## BP entropy
            # temp = node_belief * torch.log(node_belief + 1e-5)
            # node_Hvalue = -1.0 * temp.sum(dim=1)
            
            # temp = edge_belief * torch.log(edge_belief + 1e-5)
            # edge_Hvalue = -1.0 * temp.sum(dim=1)
            
            # BP_entropy_node = torch.matmul(torch.from_numpy(Ci_BP).float().to(device), node_Hvalue) 
            # BP_entropy_edge = torch.matmul(torch.from_numpy(Cij_BP).float().to(device), edge_Hvalue)
            
            # print(node_belief)
            batch_loss.append(torch.norm(Ci-Ci_BP_tensor) + torch.norm(Cij-Cij_BP_tensor))
            entropy_batch_loss.append(entropy_)
            
            
            if (i % self.batch_size == 0):
                ll_mean = torch.stack(batch_loss).mean()
                ll_mean.backward()
                optimizer.step()
                self.model.zero_grad()
                batch_loss=[]
                mean_losses.append(ll_mean.item())
                
                entropy_mean = torch.stack(entropy_batch_loss).mean()
                entropy_batch_loss = []
                entropy_mean_losses.append(entropy_mean.item())
                

        self.history["loss"].append(np.mean(mean_losses))
        self.history["entropy"].append(np.mean(entropy_mean_losses))