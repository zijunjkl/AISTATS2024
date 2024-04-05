"""

Exact inference
Authors: kkorovin@cs.cmu.edu

"""
import itertools
import numpy as np
from tqdm import tqdm
from inference.core import Inference


class ExactInference(Inference):
    """ Special case BinaryMRF implementation """
    def _safe_norm_exp(self, logit):
        # logit_orig = logit
        logit -= np.max(logit, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(keepdims=True)        
        # prob2 = np.exp(logit_orig)
        # prob2 /= prob2.sum(keepdims=True)
        return prob

    def compute_probs(self, W, bias, n):
        log_potentials = np.zeros([2]*n)
        for state in itertools.product([0, 1], repeat=n):
            state_ind = np.array(state)
            state_val = 2 * state_ind - 1
            log_potentials[state] = state_val.dot(W.dot(state_val))/2 + bias.dot(state_val)
        probs = self._safe_norm_exp(log_potentials)
        return probs

        
    def run_one(self, graph):
        W = graph.W
        b = graph.b
        n = graph.n_nodes

        # compute joint probabilities
        # array of shape [2,...,2]
        probs = self.compute_probs(W, b, n)
        # print("M1:", probs[0, :, :].sum(), probs[1, :, :].sum())
        # print("M2:", probs[:, 0, :].sum(), probs[:, 1, :].sum())
        
        adj = W.copy()
        adj[np.abs(W)>0] = 1   
        uptriangle_adj = np.triu(adj)
    
        if self.mode == "marginal":
            # select one state and compute marginal:
            marginals = np.zeros((n, 2))  # [i, 0] is P(x_i=0)
            for i in range(n):
                axes = tuple(j for j in range(n) if j != i)
                marginal = probs.sum(axis=axes)
                marginals[i] = marginal
            gt_unary_probs = marginals  # marginal probabilities for each node
            
            row, col = np.nonzero(uptriangle_adj) #
            num_edge = len(row)
            gt_pairwise_probs = np.zeros((num_edge, 4))
            for idx in range(num_edge):
                j_node = row[idx]
                i_node = col[idx]
                axes = tuple(k for k in range(n) if k != i_node and k != j_node)
                pairwise = probs.sum(axis=axes)
                # pairwise = probs.max(axis=axes)
                if i_node > j_node:
                    gt_pairwise_probs[idx,:] = np.reshape(pairwise, [1,4])
                else:
                    gt_pairwise_probs[idx,:] = np.reshape(pairwise.T, [1,4])
                # gt_pairwise_probs[idx,:,:] = gt_pairwise_probs[idx,:,:]/np.sum(gt_pairwise_probs[idx,:,:])
            
            fake_map_config = np.zeros([1,n])
            
            return gt_unary_probs

        elif self.mode == "map":
            
            # select one state and compute marginal:
            max_marginals = np.zeros((n, 2))  # [i, 0] is P(x_i=0)
            for i in range(n):
                axes = tuple(j for j in range(n) if j != i)
                max_marginal = probs.max(axis=axes)
                max_marginals[i] = max_marginal #/np.sum(max_marginal)
            gt_unary_probs = max_marginals  # max-marginal probabilities for each node
            
            row, col = np.nonzero(uptriangle_adj) #
            num_edge = len(row)
            gt_pairwise_probs = np.zeros((num_edge, 4))
            for idx in range(num_edge):
                j_node = row[idx]
                i_node = col[idx]
                axes = tuple(k for k in range(n) if k != i_node and k != j_node)
                max_pairwise = probs.max(axis=axes)
                if i_node > j_node:
                    gt_pairwise_probs[idx,:] = np.reshape(max_pairwise, [1,4])
                else:
                    gt_pairwise_probs[idx,:] = np.reshape(max_pairwise.T, [1,4])
                gt_pairwise_probs[idx,:] = gt_pairwise_probs[idx,:] #/np.sum(gt_pairwise_probs[idx,:])


            binary_ind = np.unravel_index(probs.argmax(), probs.shape)
            map_config = 2 * np.array(binary_ind) - 1
            
            num_node = n
            one_hot_map_config = np.zeros((n,2))
            for j in range(num_node):
                idx = np.argmax(gt_unary_probs[j,:])
                one_hot_map_config[j,idx] = 1
            one_hot_edge_map_config = np.zeros((num_edge, 4))
            for j in range(num_edge):
                idx = np.argmax(gt_pairwise_probs[j,:])
                one_hot_edge_map_config[j,idx] = 1
            
            return gt_unary_probs, gt_pairwise_probs, map_config, one_hot_map_config, one_hot_edge_map_config


    def run(self, graphs, verbose=False):
        self.verbose = verbose
        res = []
        graph_iterator = tqdm(graphs) if self.verbose else graphs
        for graph in graph_iterator:
            res.append(self.run_one(graph))
        return res
