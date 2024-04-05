#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:23:59 2020

@author: zijun.cui
"""

import os
import argparse
import numpy as np
from pprint import pprint
from time import time
import matplotlib.pyplot as plt

from graphical_models import construct_binary_mrf, BinaryMRF
from inference import get_algorithm
from labeling import LabelProp, LabelSG, LabelTree



def load_graphs(path):
    graphs = np.load(path, allow_pickle=True)
    return graphs


# 
low, high = 16, 16
size_range = np.arange(int(low), int(high)+1)
algo = 'exact'
num = 100
graph_struct ="path"

#struct_names = ["star", "random_tree", "powerlaw_tree", "path",
#                "cycle", "ladder", "grid",
#                "circ_ladder", "barbell", "loll", "wheel",
#                "bipart", "tripart", "fc"]

mode = 'marginal'
Verbose = True
data_mode = 'test'


#directory to save a generated dataset
# base_data_dir ='./graphical_models/datasets/'
base_data_dir ='./graphical_models/datasets-marginal/'

#whether to use previously created unlabeled graphs.
#If `none`, creates new graphs. 
#If non-`none`, should be a path from base_data_dir
unlab_graphs_path ='none'

                            
# construct graphical models
# either new-data-generation or data labeling scenario
if unlab_graphs_path == 'none' or algo == 'none':
    # create new graphs
    graphs = []
    for _ in range(num):
        # sample n_nodes from range
        n_nodes = np.random.choice(size_range)
        graphs.append(construct_binary_mrf(graph_struct, n_nodes))
else:  # both are non-None: need to load data and label it
    path = os.path.join(base_data_dir, unlab_graphs_path)
    graphs = load_graphs(path + '.npy')



# label them using a chosen algorithm
if algo in ['exact', 'bp', 'mcmc']:
    algo_obj = get_algorithm(algo)(mode)
    list_of_res = algo_obj.run(graphs, verbose=Verbose)

elif algo.startswith('label_prop'): # Propagate-from-subgraph algorithm (pt 2.2)
    # e.g. label_prop_exact_10_5
    inf_algo_name, sg_sizes = algo.split('_')[2], algo.split('_')[3:]
    sg_sizes = list(map(int, sg_sizes))
    inf_algo = get_algorithm(inf_algo_name)(mode)
    label_prop = LabelProp(sg_sizes, inf_algo, max_iter=30)
    list_of_res = label_prop.run(graphs, verbose=Verbose)

elif algo == 'label_tree': # Subgraph labeling algorithm (pt 2.1):
    lbt = LabelTree(mode)
    list_of_res = lbt.run(graphs, verbose=Verbose)

elif algo.startswith('label_sg'):
    algo_method = algo.split('_')[2]
    inf_algo_name = 'exact' # we will be using the default inf_algo
    inf_algo = get_algorithm(inf_algo_name)(mode)
    sg_labeler = LabelSG(inf_algo, algo_method)
    list_of_res = sg_labeler.run(graphs, verbose=Verbose)

elif algo == 'none':
    list_of_res = [None] * len(graphs)
else:
    raise ValueError("Labeling algorithm {algo} not supported.")


# saves to final paths if labeled, otherwise to unlab_graphs_path
# unlabeled data, save to its temporary address
count = 0
if algo == 'none':
    path = os.path.join(base_data_dir, unlab_graphs_path)
    np.save(path + '.npy', graphs, allow_pickle=True)
# otherwise the data is prepared and should be saved
else:
    for graph, res in zip(graphs, list_of_res):
        if mode == "marginal":
            res_marginal, res_map = res, None
        else:
            res_unary_marginal, res_pairwise_marginal, res_map, res_map_one_hot, res_pairwise_map_one_hot = res[0], res[1], res[2], res[3], res[4]
        # res_unary_marginal, res_pairwise_marginal, res_map, res_map_one_hot, res_pairwise_map_one_hot = res[0], res[1], res[2], res[3], res[4]
        
        # directory = os.path.join(base_data_dir, data_mode,
        #                           graph.struct, str(graph.n_nodes))
        directory = os.path.join(base_data_dir, data_mode,
                                  'loop-free', str(graph.n_nodes))
        os.makedirs(directory, exist_ok=True)
        ## for marginal
        if mode == "marginal":
            data = {"W": graph.W, "b": graph.b, "unary_marginal": res_marginal}
        # ## for MAP
        else:
            data = {"W": graph.W, "b": graph.b,
                    "unary_marginal": res_unary_marginal, "pairwise_marginal": res_pairwise_marginal, "map": res_map,
                    "unary_map_one_hot": res_map_one_hot, "pairwise_map_one_hot": res_pairwise_map_one_hot}
        count = count + 1
        t = "_".join(str(time()).split(".")) + '_'+str(count)
        path_to_graph = os.path.join(directory, t)
        np.save(path_to_graph, data)
        
print(graph_struct)
print(data_mode)
