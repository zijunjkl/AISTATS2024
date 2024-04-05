#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 21:46:27 2020

@author: zijun.cui
"""

import random
import time
from experiments.exp_helpers import get_dataset_by_name
from inference import get_algorithm
from constants import *
import numpy as np


mode = 'map' #map or marginal

## datasets
train_set_name = 'barbell_small'
model_name = 'barbell_small'


data_dir = './graphical_models/datasets/train'
test_data_dir = './graphical_models/datasets/test'
            
dataset = get_dataset_by_name(train_set_name,  data_dir,  mode)
test_data  = get_dataset_by_name(train_set_name, test_data_dir, mode)



acc_store = []

for r in range(10):
    acc_bp = 0
    acc_damp = 0
    acc_mplp = 0
    
    time_bp = 0
    time_damp = 0
    time_mplp = 0
    
    ## testing labels
    for g in test_data:
        
        mode = 'map'
        gt_map = g.map
    
        H = g.W
        bias = g.b
        N = len(H)
        
        bx= np.zeros([N, 2])
        bx[:,0] = -bias
        bx[:,1] = bias
        b_arr = np.exp(bx)
        
        adj = H.copy()
        adj[np.abs(H)>0] = 1
        
        uptriangle_adj = np.triu(adj)
        
        E = np.count_nonzero(uptriangle_adj)
        edge_appear_prob = np.ones([N,N])*(N-1)/E
        edge_appear_prob = edge_appear_prob*adj
        
        edge_appear_prob_BP = adj.copy()
        
        
        
        bp = get_algorithm("bp")(mode)
        
        graph = [adj, b_arr, H, gt_map, np.arange(N), 0, 0]
        start = time.time()
        lbp_marginal, lbp_map, acc_arr_bp, dist_arr_bp, diff_arr_bp = bp.run(graph, use_log=False, verbose=False)
        acc_bp += dist_arr_bp[-1]
        time_bp += time.time() - start
        
    
        TRWbp = get_algorithm("TRWbp")(mode)
        graph = [adj, b_arr, H, gt_map, np.arange(N), edge_appear_prob, 0.5]
        start = time.time()
        trw_fix_marginal, trw_fix_map, acc_arr_damp, dist_arr_damp, diff_arr_damp = TRWbp.run(graph, use_log=False, verbose=False)
        acc_damp += dist_arr_damp[-1]
        time_damp += time.time() - start
        
        
        MPLP = get_algorithm("MPLP")(mode)
        graph = [adj, b_arr, H, gt_map, np.arange(N)]
        start = time.time()
        mplp_marginal, mplp_map, acc_arr_mplp, dist_arr_mplp, diff_arr_mplp = MPLP.run(graph, use_log=False, verbose=False)
        acc_mplp += dist_arr_mplp[-1]
        time_mplp += time.time() - start
    
    print("Accuracy BP = %f, TRWMP = %f, MPLP = %f"%(acc_bp/100, acc_damp/100, acc_mplp/100))
    print("Computing time BP = %f, TRWMP = %f, MPLP = %f"%(time_bp/100, time_damp/100, time_mplp/100))
