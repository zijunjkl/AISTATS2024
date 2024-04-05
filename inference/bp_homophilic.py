"""

Approximate inference using Belief Propagation
Here we can rely on some existing library,
for example https://github.com/mbforbes/py-factorgraph
Authors: lingxiao@cmu.edu
         kkorovin@cs.cmu.edu
"""

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import time
from inference.core import Inference
from scipy.stats import entropy

def compute_potential_2(state, adj, unary, H):
    state = state.astype(int)
    adj = np.triu(adj)
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

class BeliefPropagation(Inference):
    """
    A special case implementation of BP
    for binary MRFs.
    Exact BP in tree structure only need two passes,
    LBP need multiple passes until convergene. 
    """

    def _safe_norm_exp(self, logit):
        logit -= np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def _safe_divide(self, a, b):
        '''
        Divies a by b, then turns nans and infs into 0, so all division by 0
        becomes 0.
        '''
        c = a / b
        # c[c == np.inf] = 0.0
        # c = np.nan_to_num(c)
        return c


    def run_one(self, adj, b_arr, H, gt_test_label, test_idx, damping, sequential):
        # GNN - BP  
        # follow the procedure of GNN-BP
        if self.mode == "marginal": # not using log
            sumOp = np.sum
        else:
            sumOp = np.max

        max_iters = 200
        epsilon = 1e-7 # determines when to stop
        error_arr = []
        time_arr = []
        acc_arr = []
        dist = []
        potential = []
        k = b_arr.shape[1]
        row, col = np.where(adj)
        n_V, n_E = len(b_arr), len(row)
        # create index dict
        degrees = np.sum(adj != 0, axis=0) #incoming links number of nonzeros for each column
        index_bases = np.zeros(n_V, dtype=np.int64)
        for i in range(1, n_V): 
            index_bases[i] = index_bases[i-1] + degrees[i-1]

        in_neighbors = {i:[] for i in range(n_V)}
        out_neighbors = {i:[] for i in range(n_V)}
        for i,j in zip(row,col): 
            out_neighbors[i].append(j) # outgoing: i send message to j
            in_neighbors[j].append(i) #incoming: j receive message from i
        in_neighbors = {k: sorted(v) for k, v in in_neighbors.items()}
        
        ordered_nodes = np.arange(n_V) #no order bias
        
        if sequential == 1:
            ordered_nodes = np.arange(n_V)
            
        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        #messages = np.ones([n_E, k])
        messages = np.random.rand(n_E, k)
        
        
        
        # calculate marginal or map
        probs = np.zeros([n_V, k])
        for i in range(n_V):
            probs[i] = b_arr[i,:] 
            for j in in_neighbors[i]:
                probs[i] *= messages[index_bases[i]+in_neighbors[i].index(j)] 
                
        probs = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))

        converged = False
        for count in range(max_iters):
            # print('iteration {}'.format(count))
            start = time.time()
            # save old message for checking convergence
            old_messages = messages.copy()
            old_probs = probs.copy()
            # update messages
            for i in ordered_nodes:
                # print(i)
                in_neigh = in_neighbors[i] #incoming messages
                # calculate all the in-coming messages to i product (log)
                in_message_prod = np.array([1]*k, dtype='float64')
                for j in in_neigh:
                    in_message_prod *= old_messages[index_bases[i]+in_neighbors[i].index(j)]
                    
                # send message from i to neighbors
                out_neigh = out_neighbors[i] 
                
                if sequential == 1:
                    out_neigh_seq = []
                    order_i = np.where(ordered_nodes == i)[0]
                    for j in out_neigh:
                        order_j = np.where(ordered_nodes == j)[0]
                        if order_j > order_i:
                            out_neigh_seq.append(j)
                    out_neigh = out_neigh_seq.copy()
                    
                unary = b_arr[i,:]
                for j in out_neigh:
                    old_message_jtoi = old_messages[index_bases[i]+in_neighbors[i].index(j)]
                    message_itoj = self._safe_divide(in_message_prod, old_message_jtoi)

                    
                    eta_ij = H[i,j]
                    phi_ij = np.exp([[eta_ij, -eta_ij],[-eta_ij, eta_ij]])

                    messages[index_bases[j]+in_neighbors[j].index(i)] = sumOp(message_itoj.reshape(1,k,1)*phi_ij*(unary.reshape(-1,1)), axis=1)
                
                if sequential == 1:
                    messages /= messages.sum(axis=1, keepdims=True)
                    old_messages = messages.copy()
                    
            
            if sequential == 1:
                ordered_nodes = n_V - 1 - ordered_nodes
            
            
            if damping == 1:
                # print('apply message damping with 0.5')
                # damping
                messages = np.exp(0.5*np.log(messages) + 0.5*np.log(old_messages))
                # messages = np.exp(np.log(messages))
            
            
            # normalize the messages
            messages /= messages.sum(axis=1, keepdims=True)
            
            # check convergence 
            # calculate marginal or map
            probs = np.zeros([n_V, k])
            for i in range(n_V):
                probs[i] = b_arr[i,:]
                for j in in_neighbors[i]:
                    probs[i] *= messages[index_bases[i]+in_neighbors[i].index(j)] 
                    
            probs = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))
                
            error = (probs - old_probs)**2
            
            if len(error):
                error = np.sum(error, 1)
                error = error.mean()
            else:
                error = 0.
            
            end = time.time()
            error_arr.append(error)
            time_arr.append((end-start))
            
            bp_labels = np.zeros(n_V)
            for ii in np.arange(n_V):  
                bp_labels[ii] = np.argmax(probs[ii,:])
            
            
            if self.mode == "map":
                # potential.append(np.log(compute_potential_2(bp_labels, adj, b_arr, H)))
                bp_labels[bp_labels==0] = -1
                bp_test = bp_labels
                if np.array_equal(gt_test_label, bp_test):
                    test_acc = 1
                else:
                    test_acc = 0
                acc_arr.append(test_acc)
                dist.append(np.mean(bp_labels == gt_test_label))
            elif self.mode == 'marginal':
                # KL-divergence
                kl = 0
                for ii in np.arange(n_V):
                    kl = kl + entropy(probs[ii,:], gt_test_label[ii,:])
                acc_arr.append(kl/n_V)
            
            # dist.append(np.linalg.norm((gt_test_label-bp_test), ord=2))
            
            if error < epsilon: 
                converged = True
                break

        # calculate marginal or map
        probs = np.zeros([n_V, k])
        for i in range(n_V):
            probs[i] = b_arr[i,:]
            for j in in_neighbors[i]:
                probs[i] *= messages[index_bases[i]+in_neighbors[i].index(j)] 
        # normalize
        if self.mode == 'marginal':
            results = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))

        if self.mode == 'map':
            results = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))
        
        bp_labels = np.zeros(n_V)
        for ii in np.arange(n_V):  
            bp_labels[ii] = np.argmax(results[ii,:])
        
        
        if self.mode == "map":
            # potential.append(np.log(compute_potential_2(bp_labels, adj, b_arr, H)))
            bp_labels[bp_labels==0] = -1
            bp_test = bp_labels
            if np.array_equal(gt_test_label, bp_test):
                test_acc = 1
            else:
                test_acc = 0
            acc_arr.append(test_acc)
            dist.append(np.mean(bp_labels == gt_test_label))    
        elif self.mode == 'marginal':
            # KL-divergence
            kl = 0
            for ii in np.arange(n_V):
                kl = kl - np.log10(entropy(results[ii,:], gt_test_label[ii,:])+1e-10)
                # kl = kl + entropy(results[ii,:], gt_test_label[ii,:])
            acc_arr.append(kl/n_V)

        # dist.append(np.linalg.norm((gt_test_label-bp_test), ord=2))
            
        
        return results, bp_labels, acc_arr, dist, error_arr

    def run(self, graph, use_log=False, verbose=False):
        self.verbose = verbose
        
        output = self.run_one(graph[0], graph[1], graph[2], graph[3], graph[4], 1, graph[6])
        results = output[0]
        map = output[1]
        acc = output[2]
        dist = output[3]
        error = output[4]

        return results, map, acc, dist, error


if __name__ == "__main__":
    bp = BeliefPropagation("marginal")
    
