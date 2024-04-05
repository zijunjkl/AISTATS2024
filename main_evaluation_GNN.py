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
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from experiments.exp_helpers import get_dataset_by_name
from inference import get_algorithm
from constants import *
import numpy as np
import scipy.io as sio


def Measure_ACC_Dist(train_true_labels, gnn_labels):
    n_sample = train_true_labels.shape[0]
    acc = 0
    dist = 0
    ele_acc = []
    for i in np.arange(n_sample):
        if np.array_equal(train_true_labels[i,:], gnn_labels[i,:]):
            acc = acc + 1
        ele_acc.append(np.mean(gnn_labels[i,:] == train_true_labels[i,:]))

    
    return acc/n_sample, np.mean(ele_acc)

# Loss computer objects that let us save a little on creating objects----------
class CrossEntropyComputer:
    def __init__(self):
        self.computer = nn.BCELoss()

    def __call__(self, output_probs, targets):
        return self.computer(output_probs, targets)


class CrossEntropyMAPComputer:
    def __init__(self):
        self.computer = nn.BCELoss() #binary cross entropy

    def __call__(self, output_probs, targets):
        return self.computer(output_probs[:, 1], targets)
    

mode = 'map' # marginal or map


train_set_name = 'barbell_small'
model_name = 'barbell_small'
model_path = './Node-GNN-models/9/barbell_small'




data_dir = './graphical_models/datasets/train'
test_data_dir = './graphical_models/datasets/test'
            
dataset = get_dataset_by_name(train_set_name,  data_dir,  mode)
test_data  = get_dataset_by_name(train_set_name, test_data_dir, mode)

hidden_unit_message_dim = 64
hidden_unit_readout_dim = 64
T = 10

epochs = 15

repeats = 5
store_repeats_acc = []

test_true_labels = []
for g in test_data:
    test_true_labels.append(g.map)
test_true_labels = np.array(test_true_labels)

train_true_labels = []
for g in dataset:
    train_true_labels.append(g.map)
train_true_labels = np.array(train_true_labels)
    

for rr in range(repeats):
    
    ## SoTA
    learning_rate = 1e-3
    n_hidden_states = 5
    message_dim_P = 5
    gnn_constructor = get_algorithm('gnn_inference')
    gnn_inference = gnn_constructor(mode, n_hidden_states, message_dim_P, \
                                    hidden_unit_message_dim, hidden_unit_readout_dim, T, sparse=True)
        
    optimizer = Adam(gnn_inference.model.parameters(), lr=learning_rate)
    
    
    if mode == 'marginal':
        criterion = CrossEntropyComputer()
    else:
        criterion = CrossEntropyMAPComputer()
            
    
    best_loss = 1e5
    
        
    train_acc_arr= []
    test_acc_arr = []
    best_test_ele_acc = -100
    best_test_acc = -100
    for epoch in range(epochs):
        
        if epoch % 3 == 0:
            learning_rate = learning_rate*0.8
        optimizer = Adam(gnn_inference.model.parameters(), lr=learning_rate)
        gnn_inference.train(dataset, optimizer, criterion, DEVICE)
        losses = gnn_inference.history["loss"]
        # print(losses[-1])
        
        # testing accuracy
        gnn_res = gnn_inference.run(test_data, DEVICE)
        gnn_labels = []
        for graph_res in gnn_res:
            gnn_labels.append(list(-1 if m[0]>m[1] else +1 for m in graph_res))
        gnn_labels = np.array(gnn_labels)
        gnn_accuracy, gnn_ele_acc = Measure_ACC_Dist(test_true_labels, gnn_labels)
        
        # print('\ntraining loss=%f, training_acc = %f, test_acc = %f'%(losses[-1],gnn_accuracy_train, gnn_accuracy))
        print('test_acc = %f'%(gnn_ele_acc))
        
        if gnn_ele_acc > best_test_ele_acc:
            best_test_acc = gnn_accuracy
            best_test_ele_acc = gnn_ele_acc
            #gnn_inference.save_model(model_path)
            
        train_acc_arr.extend([gnn_accuracy])
        test_acc_arr.extend([gnn_ele_acc])
    
    print('\n')
    print(train_set_name)
    print('repeat = %d, best testing_acc = %f'%(rr, best_test_ele_acc))
    store_repeats_acc.append(best_test_ele_acc)
    
print('average testing_acc = %f, std = %f'%(np.mean(store_repeats_acc), np.std(store_repeats_acc)))



