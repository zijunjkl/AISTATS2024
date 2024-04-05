#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 21:46:27 2020

@author: zijun.cui
"""

import os
import argparse
from time import time
import torch
import itertools
import random
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import networkx as nx
from experiments.exp_helpers import get_dataset_by_name
from inference import get_algorithm
from constants import *
import numpy as np
from graphical_models import BinaryMRF
import scipy.io as sio
from scipy.stats import entropy



def Measure_ACC(dataset, mpnn_node_marginals):   
    n_sample = len(dataset)
    ele_acc = []
    for i in np.arange(n_sample):
        est_map = list(-1 if m[0]>m[1] else +1 for m in mpnn_node_marginals[i])        
        est_map = np.asarray(est_map)
        
        ## element-wise accuracy
        true_map = dataset[i].map
        ele_acc.append(np.mean(est_map == true_map))
        # if np.array_equal(true_map, est_map):
        #     test_acc = 1
        # else:
        #     test_acc = 0
        # ele_acc.append(test_acc)
                
    return np.mean(ele_acc)




def Measure_ACC_edge(dataset, mpnn_edge_marginals):   
    ''' 
    convert to unary MAP 
    '''
    n_sample = len(dataset)
    ele_acc = []
    ele_inconsis = []
    for i in np.arange(n_sample):
        graph = dataset[i]
        J = graph.W
        adj = J.copy()
        adj[np.abs(J)>0] = 1   
        uptriangle_adj = np.triu(adj)
        row, col = np.nonzero(uptriangle_adj) 
        
        est_ = mpnn_edge_marginals[i]
        edge_row = np.zeros((len(row), 2))
        # edge_row[:,0] = est_[:,0] + est_[:,1]
        # edge_row[:,1] = est_[:,2] + est_[:,3]
        # ## max marginals for map
        temp = est_[:,[0,1]]
        edge_row[:,0] = np.max(est_[:,[0,1]], axis=1, keepdims=True)[:,0]
        edge_row[:,1] = np.max(est_[:,[2,3]], axis=1, keepdims=True)[:,0]
        
        edge_col = np.zeros((len(row), 2))
        # edge_col[:,0] = est_[:,0] + est_[:,2]
        # edge_col[:,1] = est_[:,1] + est_[:,3]
        # ## max marginals for map
        edge_col[:,0] = np.max(est_[:,[0,2]], axis=1, keepdims=True)[:,0]
        edge_col[:,1] = np.max(est_[:,[1,3]], axis=1, keepdims=True)[:,0]
        
        
        # node_belief_row = np.zeros((num_node, 2))
        # node_belief_edge = np.zeros((num_node, 2))
        # node_belief_row[row] = edge_row
        # node_belief_edge[col] = edge_col
        
        num_node = J.shape[0]
        true_ = dataset[i].map
        acc = 0
        inconsis_arr = np.zeros(num_node)
        for j in np.arange(num_node):
            exist_flag = 0
            est_map_j = []
            idx = []
            est_belief_j = []
            ## extract from row
            idx = np.where(row == j)[0]
            if len(idx) > 0:
                est_belief_j = edge_row[idx]
                exist_flag = 1
                
            ## extract from col
            idx = np.where(col == j)[0]
            if len(idx) > 0:
                temp = edge_col[idx].reshape([len(idx),2])
                if exist_flag == 1:
                    est_belief_j = np.concatenate((est_belief_j, temp))
                elif exist_flag == 0:
                    est_belief_j = temp
            
            ## extract MAP
            est_map_j = np.argmax(est_belief_j, axis=1)
            est_map_j[np.where(est_map_j==0)[0]] = -1
            
            ## randomly choose one 
            rnd_idx = random.choice(range(0, len(est_map_j)))
            if est_map_j[rnd_idx] == true_[j]:
                acc = acc + 1
            
            ## count the number of inconsistent estimation for node j
            inconsis_arr[j] = len(np.where(est_map_j != true_[j])[0]) 
            
        ele_acc.append(acc/num_node)
        ele_inconsis.append(inconsis_arr)
        
    return np.mean(ele_acc), np.mean(ele_inconsis, axis = 0)


def Measure_ACC_Energy_delta(dataset, mpnn_node_marginals, mpnn_edge_marginals, test_true_node_marginals, test_true_edge_marginals):
    n_sample = len(dataset)
    ele_acc = []
    energy = []
    for i in np.arange(n_sample):
        est_map = list(-1 if m[0]>m[1] else +1 for m in mpnn_node_marginals[i])
        unary_probs = mpnn_node_marginals[i]
        pairwise_probs = mpnn_edge_marginals[i]
        
        est_map = np.asarray(est_map)
        # unary_probs =  np.asarray(unary_probs) - test_true_node_marginals[i]
        # pairwise_probs =  np.asarray(pairwise_probs) - test_true_edge_marginals[i]
        
        # if Integer == True:
        num_node = unary_probs.shape[0]
        for j in range(num_node):
            idx = np.argmax(unary_probs[j,:])
            unary_probs[j,:] = np.zeros([1,2])
            unary_probs[j,idx] = 1
        num_edge = pairwise_probs.shape[0]
        for j in range(num_edge):
            idx = np.argmax(pairwise_probs[j,:])
            pairwise_probs[j,:] = np.zeros([1,4])
            pairwise_probs[j,idx] = 1
        unary_probs =  np.asarray(unary_probs) - test_true_node_marginals[i]
        pairwise_probs =  np.asarray(pairwise_probs) - test_true_edge_marginals[i]
        
        
        ## element-wise accuracy
        true_map = dataset[i].map
        ele_acc.append(np.mean(est_map == true_map))
        
        ## energy
        bias = dataset[i].b
        num_nodes = len(bias)
        b_arr = np.ones([num_nodes,2])
        b_arr[:,0] = -bias
        b_arr[:,1] = bias
        b_arr = np.exp(b_arr)
        average_energy_marginal = np.sum(unary_probs*np.log(b_arr))
        
        H = dataset[i].W
        rows, cols = np.where(np.triu(H))
        num_E = len(rows)
        average_energy_pair = 0
        for ee in range(num_E):
            node_i = rows[ee]
            node_j = cols[ee]
            eta_ij = H[node_i, node_j]
            H_ij = np.exp([[eta_ij, -eta_ij],[-eta_ij, eta_ij]])
            pairwise_ = np.reshape(pairwise_probs[ee], [2,2])
            average_energy_pair = average_energy_pair + np.sum(pairwise_*np.log(H_ij))
        
        total_energy = -average_energy_pair - average_energy_marginal
        energy.append(total_energy)
        
    return np.mean(ele_acc), np.mean(energy), energy

def Measure_bound_tightness(dataset, mpnn_node_marginals, mpnn_edge_marginals, test_true_node_marginals, test_true_edge_marginals, entropy_):
    _,_,energy_delta = Measure_ACC_Energy_delta(dataset, mpnn_node_marginals, mpnn_edge_marginals, test_true_node_marginals, test_true_edge_marginals)
    epsilon = 0.1
    bound = np.asarray(entropy_)*0.0001 - energy_delta
    
    return np.mean(bound)



mode = 'map' #map or marginal

init_model_path = './G-MPNN-map-models/pre-train-GTMAP-GMPNN-map-loop-free_9'


## datasets

train_set_name = 'barbell_small'
model_path = './fine-tune-entropy-GMPNN-map-barbell_small'


data_dir = './graphical_models/datasets/train'
test_data_dir = './graphical_models/datasets/test'
            
dataset = get_dataset_by_name(train_set_name,  data_dir,  mode)
test_data  = get_dataset_by_name(train_set_name, test_data_dir, mode)


hidden_unit_message_dim = 256
hidden_unit_readout_dim = 256
T = 10


## training labels
train_true_maps = []
train_true_node_marginals = []
train_true_edge_marginals = []
for g in dataset:
    train_true_maps.append(g.map)
    train_true_node_marginals.append(g.unary_map_onehot)
    train_true_edge_marginals.append(g.pairwise_map_onehot)
train_true_maps = np.array(train_true_maps)
train_true_node_marginals = np.array(train_true_node_marginals)
train_true_edge_marginals = np.array(train_true_edge_marginals)
    

## testing labels
test_true_maps = []
test_true_node_marginals = []
test_true_edge_marginals = []
for g in test_data:
    test_true_maps.append(g.map)
    test_true_node_marginals.append(g.unary_map_onehot)
    test_true_edge_marginals.append(g.pairwise_map_onehot)
test_true_maps = np.array(test_true_maps)
test_true_node_marginals = np.array(test_true_node_marginals)
test_true_edge_marginals = np.array(test_true_edge_marginals)




store_repeat_acc = []
repeat = 10

for rr in range(repeat):
    
    
    ## node-MPNN: without pretraining
    n_hidden_states = 2
    message_dim_P = 2
    gnn_constructor = get_algorithm('mpnn_general_form_gt')
    method = 'MPNNgeneral_gt'
    mpnns = gnn_constructor(mode, method, n_hidden_states, message_dim_P, \
                                    hidden_unit_message_dim, hidden_unit_readout_dim, \
                                        T, load_path =init_model_path, sparse=True)

    

    temp = dataset[0].W
    node_dim = temp.shape[0]
    row, col = np.nonzero(np.triu(temp))
    edge_dim = len(row)

    adj_matrix = temp.copy()
    adj_matrix[np.abs(temp)>0] = 1
    
    connectivity = np.sum(adj_matrix, axis=1)
    Ci_BP = 1 - connectivity
    Cij_BP = np.ones((1, edge_dim))      
            
    # print('Ci BP')
    # print(Ci_BP)
    # print('Cij_BP')
    # print(Cij_BP)
    
    best_loss = 1e5
    
    # ## training accuracy 
    # mpnn_node_beliefs, mpnn_edge_beliefs, entropy, Ci_train, Cij_train = mpnns.run(dataset, 'both', 200, DEVICE)
    # mpnn_node_acc_train = Measure_ACC(dataset, mpnn_node_beliefs)    
    # mpnn_edge_acc_train, mpnn_edge_inconsis_train = Measure_ACC_edge(dataset, mpnn_edge_beliefs)
    
    ## testing accuracy
    mpnn_node_beliefs_test, mpnn_edge_beliefs_test, entropy_, Ci_test, Cij_test, _ = mpnns.run(test_data, 'both', 200,  DEVICE)
    mpnn_node_acc_test = Measure_ACC(test_data, mpnn_node_beliefs_test)
    mpnn_node_tight_test = Measure_bound_tightness(test_data, mpnn_node_beliefs_test, mpnn_edge_beliefs_test, test_true_node_marginals, test_true_edge_marginals, entropy_)
    # mpnn_edge_acc_test, mpnn_edge_inconsis_test = Measure_ACC_edge(test_data, mpnn_edge_beliefs_test)
    print(mpnn_node_tight_test)
    print(train_set_name)
    
    train_ACC_node = []
    test_ACC_node = []
    # train_ACC_edge = []
    # test_ACC_edge = []
    
    # print(train_set_name)
    # print('initial model performance:')
    # print('testing_node_ACC = %f'%(mpnn_node_acc_test))
    # print('training_edge_ACC = %f, testing_edge_ACC = %f'%(mpnn_edge_acc_train, mpnn_edge_acc_test))
    
    
    # train_ACC_node.append(mpnn_node_acc_train)
    # test_ACC_node.append(mpnn_node_acc_test)
    # train_ACC_edge.append(mpnn_edge_acc_train)
    # test_ACC_edge.append(mpnn_edge_acc_test)        
    
    # print('\ntraining with entropy start')
    
    '''
    training 
    '''
     
    ## number of epochs
    epochs = 50
    
    # learning rate
    learning_rate = 1e-5
    
    train_entropy = []
    test_entropy = []
    
    best_acc = mpnn_node_acc_test
    
    
    T_train = 200
    T_test = 200


    for epoch in range(epochs):
        # if epoch != 0 and epoch % 5 == 0:
        #     learning_rate = learning_rate*0.85
            
        ## train of mpnn_node with map annotations
        mpnn_optimizer = Adam(mpnns.model.parameters(), lr=learning_rate)
        mpnns.train_free_energy(dataset, mpnn_optimizer,Ci_BP, Cij_BP, T_train, DEVICE)
        losses = mpnns.history["loss"]
        cMSEs = mpnns.history["cMSE"]
        
        # print('\nepoch=%d'%epoch)
        # print(train_set_name)
        # print('\nloss = %f, cMSE = %f'%(losses[-1], cMSEs[-1]))
    
        # ## training accuracy 
        # mpnn_node_beliefs, mpnn_edge_beliefs, entropy_train, Ci_train, Cij_train = mpnns.run(dataset, 'both', DEVICE)
        # mpnn_node_acc_train = Measure_ACC(dataset, mpnn_node_beliefs)
        # # mpnn_edge_acc_train, _ = Measure_ACC_edge(dataset, mpnn_edge_beliefs)
        # Ci_train = np.asarray(Ci_train)
        # Cij_train = np.asarray(Cij_train)
        
        ## testing accuracy
        mpnn_node_beliefs_test, mpnn_edge_beliefs_test, entropy_test, Ci_test, Cij_test, _ = mpnns.run(test_data, 'both',T_test, DEVICE)
        mpnn_node_acc_test = Measure_ACC(test_data, mpnn_node_beliefs_test)
        # mpnn_edge_acc_test, _ = Measure_ACC_edge(test_data, mpnn_edge_beliefs_test)
        Ci_test = np.asarray(Ci_test)
        Cij_test = np.asarray(Cij_test)
        
        # # # print('===========================')
        # print('testing_node_ACC = %f'%(mpnn_node_acc_test))
        # print('training_node_ACC = %f'%(mpnn_node_acc_train))
        # print('Average Ci (testing set)')
        # print(np.mean(Ci_test, axis=0))
        # print('Average Cij (testing set)')
        # print(np.mean(Cij_test, axis=0))
        
        # train_ACC_node.append(mpnn_node_acc_train)
        # train_ACC_edge.append(mpnn_edge_acc_train)
        test_ACC_node.append(mpnn_node_acc_test)
        # test_ACC_edge.append(mpnn_edge_acc_test)
        # train_entropy.append(np.mean(entropy_train))
        test_entropy.append(np.mean(entropy_test))
        
        if mpnn_node_acc_test > best_acc:
            # best_node_acc_train = mpnn_node_acc_train
            best_node_acc_test = mpnn_node_acc_test
            # best_edge_acc_train = mpnn_edge_acc_train
            # best_edge_acc_test = mpnn_edge_acc_test
            
            best_acc = mpnn_node_acc_test
            # mpnns.save_model(model_path)
            
        # print('\nBest performance:')
        # print('testing_node_ACC = %f'%(best_acc))
        # print(model_path)
    
    
    
    # print('\nBest performance:')
    print('repeat = %d, testing_node_ACC = %f'%(rr, best_acc))
    store_repeat_acc.append(best_acc)
    
    
print(model_path)
print('average testing_acc = %f, std = %f'%(np.mean(store_repeat_acc), np.std(store_repeat_acc)))


