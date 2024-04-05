"""

Unit tests for inference objects
Authors: kkorovin@cs.cmu.edu

"""

import os
from time import time
import numpy as np
from scipy.stats import pearsonr
import torch
import unittest

from inference import get_algorithm
from graphical_models import construct_binary_mrf 


class TestInference(unittest.TestCase):
    def setUp(self):
        self.graph = construct_binary_mrf("star", n_nodes=5, shuffle_nodes=False)
        self.graph2 = construct_binary_mrf("fc", n_nodes=5)

    def _test_exact_probs(self):
        graph = construct_binary_mrf("fc", 3)
        # compute probs:
        probs = np.zeros((2,2,2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    state = 2*np.array([i,j,k])-1
                    probs[i, j, k] = state.dot(graph.W.dot(state)) + graph.b.dot(state)
        probs = np.exp(probs)
        probs /= probs.sum()

        exact = get_algorithm("exact")("marginal")
        exact_probs = exact.compute_probs(graph.W, graph.b, graph.n_nodes)
        assert np.allclose(probs, exact_probs)

    def _test_exact(self):
        # check probs computation
        exact = get_algorithm("exact")("marginal")
        print("exact")
        print(exact.run([self.graph]))
        #exact.reset_mode("map")
        #print(exact.run([self.graph]))

    def _test_tree_bp(self):
        bp = get_algorithm("tree_bp")("marginal")
        res = bp.run([self.graph])
        print("tree_bp")
        print(res)

    def _test_bp(self):
        # BP fails on n=2 and n=3 star (on fully-conn n=3 - ok)
        bp = get_algorithm("bp")("marginal")
        res = bp.run([self.graph], use_log=True)
        print("bp")
        print(res)

    def _test_bp_nonsparse(self):
        # BP fails on n=2 and n=3 star (on fully-conn n=3 - ok)
        bp = get_algorithm("bp_nonsparse")("marginal")
        res = bp.run([self.graph], use_log=True)
        print("bp nonsparse")
        print(res)

    def _test_mcmc(self):
        mcmc = get_algorithm("mcmc")("marginal")
        res = mcmc.run([self.graph2])
        print("mcmc")
        print(res)

    def _test_gnn(self):
        # print("Testing GNN constructor")

        # GGNN parmeters
        graph = self.graph
        n_nodes = graph.W.shape[0]
        n_hidden_states = 5
        message_dim_P = 5
        hidden_unit_message_dim = 64 
        hidden_unit_readout_dim = 64
        T = 10
        learning_rate = 1e-2
        epochs = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        gnn_constructor = get_algorithm("gnn_inference")
        exists = os.path.isfile('pretrained/gnn_model.pt')
        if(exists):
            gnn_inference = gnn_constructor('marginal', n_nodes, n_hidden_states, 
                message_dim_P,hidden_unit_message_dim, hidden_unit_readout_dim, T,'pretrained/gnn_model.pt')
            
            out = gnn_inference.run(graph,device)
            #print('gnn')
            #print(out)
        else:
            print('pretrained model needed')

    def test_exact_against_bp(self):
        n_trials = 100

        bp = get_algorithm("bp")("marginal")
        tree_bp = get_algorithm("tree_bp")("marginal")
        bp_n = get_algorithm("bp_nonsparse")("marginal")
        exact = get_algorithm("exact")("marginal")

        graphs = []
        for trial in range(n_trials):
            graph = construct_binary_mrf("random_tree", n_nodes=8, shuffle_nodes=True)
            graphs.append(graph)
        r1 = exact.run(graphs)
        r2 = bp.run(graphs)
        r3 = bp_n.run(graphs)
        r4 = tree_bp.run(graphs)

        v1, v2, v3, v4 = [], [], [], []
        for graph_res in r1:
            v1.extend([node_res[1] for node_res in graph_res])
        for graph_res in r2:
            v2.extend([node_res[1] for node_res in graph_res])
        for graph_res in r3:
            v3.extend([node_res[1] for node_res in graph_res])
        for graph_res in r4:
            v4.extend([node_res[1] for node_res in graph_res])

        corr_bp = pearsonr(v1, v2)
        corr_bpn = pearsonr(v1, v3)
        corr_treebp = pearsonr(v1, v4)
        print("Correlation between exact and BP:", corr_bp[0])
        print("Correlation between exact and BP nonsparse:", corr_bpn[0])
        print("Correlation between exact and tree BP:", corr_treebp[0])

    def _test_exact_against_mcmc(self):
        sizes = [5, 10, 15]
        n_samples = [500, 1000, 2000, 5000, 10000]
        n_trials = 100

        mcmc = get_algorithm("mcmc")("marginal")
        exact = get_algorithm("exact")("marginal")

        def get_exp_data(n_trials, n_nodes):
            graphs = []
            for trial in range(n_trials):
                graph = construct_binary_mrf("fc", n_nodes=n_nodes, shuffle_nodes=True)
                graphs.append(graph)
            return graphs

        for size in sizes:
            graphs = get_exp_data(n_trials, size)
            exact_res = exact.run(graphs)
            for n_samp in n_samples:
                mcmc_res = mcmc.run(graphs, n_samp)
                v1, v2  = [], []
                for graph_res in mcmc_res:
                    v1.extend([node_res[1] for node_res in graph_res])
                for graph_res in exact_res:
                    v2.extend([node_res[1] for node_res in graph_res])

                corr_mcmc = pearsonr(v1, v2)
                print("{},{}: correlation between exact and MCMC: {}".format(size, n_samp, corr_mcmc[0]))

    def _test_mcmc_runtimes(self):
        sizes = [5, 15, 50, 500, 1000]
        n_samples = [500, 1000, 2000]
        n_trials = 10

        mcmc = get_algorithm("mcmc")("marginal")
        exact = get_algorithm("exact")("marginal")

        def get_exp_data(n_trials, n_nodes):
            graphs = []
            for trial in range(n_trials):
                graph = construct_binary_mrf("fc", n_nodes=n_nodes, shuffle_nodes=True)
                graphs.append(graph)
            return graphs

        for size in sizes:
            graphs = get_exp_data(n_trials, size)
            for n_samp in n_samples:
                t0 = time()
                mcmc_res = mcmc.run(graphs, n_samp)
                t = time() - t0
                print("{},{}: {} seconds per graph".format(size, n_samp, t/10))

if __name__ == "__main__":
    unittest.main()
