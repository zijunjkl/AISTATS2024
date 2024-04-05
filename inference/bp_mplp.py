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
        potential *= H[state[ii], state[jj]]
    
    return potential


class BeliefPropagation_MPLP(Inference):
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


    def run_one(self, adj, b_arr, H, gt_test_label, test_idx):
        # MPLP
        # for max-product only 
        # all the calculation is in log-space
        
        log_b_arr = np.log(b_arr+1e-3) # in log-space

        max_iters = 200
        epsilon = 1e-7 # determines when to stop
        error_arr = []
        time_arr = []
        acc_arr = []
        dist = []
        potential = []
        num_states = b_arr.shape[1]
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
            
        # init messages based on graph structure (E, 2)
        # messages are ordered (out messages)
        #messages = np.zeros([n_E, num_states])
        messages = np.random.rand(n_E, num_states)
        
        
        # calculate marginal or map
        probs = np.zeros([n_V, num_states])
        for i in range(n_V):
            probs[i] = log_b_arr[i,:] 
            for j in in_neighbors[i]:
                probs[i] += messages[index_bases[i]+in_neighbors[i].index(j)] #incoming message send from j to i
            
        probs = self._safe_norm_exp(probs)
        # probs = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))

        converged = False
        for count in range(max_iters):
            # print('iteration {}'.format(count))
            start = time.time()
            # save old message for checking convergence
            old_messages = messages.copy()
            old_probs = probs.copy()
            # update messages
            for ee in range(n_E):
                i = row[ee]
                j = col[ee]
                if j > i:
                    in_neigh_i = in_neighbors[i]
                    in_neigh_j = in_neighbors[j]
                    
                    in_message_prod_i = np.array([0]*num_states, dtype='float64')
                    for k in in_neigh_i:
                        if k != j:
                            in_message_prod_i += messages[index_bases[i]+in_neighbors[i].index(k)]
                            
                    in_message_prod_j = np.array([0]*num_states, dtype='float64')
                    for k in in_neigh_j:
                        if k != i:
                            in_message_prod_j += messages[index_bases[j]+in_neighbors[j].index(k)]
                    
                    eta_ij = H[i,j]
                    phi_ij = np.array([[eta_ij, -eta_ij],[-eta_ij, eta_ij]])
                    phi_ji = phi_ij.copy()
                    
                    ## delta i to j
                    delta_itoj = np.max(in_message_prod_i.reshape(1,num_states,1)+phi_ij+log_b_arr[i,:].reshape(-1,1), axis=1)/2
                    delta_itoj = delta_itoj - (in_message_prod_j+log_b_arr[j,:])/2
                    
                    ## delta j to i
                    delta_jtoi = np.max(in_message_prod_j.reshape(1,num_states,1)+phi_ji+log_b_arr[j,:].reshape(-1,1), axis=1)/2
                    delta_jtoi = delta_jtoi - (in_message_prod_i+log_b_arr[i,:])/2
                    
                    messages[index_bases[j]+in_neighbors[j].index(i)] = delta_itoj
                    messages[index_bases[i]+in_neighbors[i].index(j)] = delta_jtoi

            # normalize the messages
            # messages = self._safe_norm_exp(messages)
            # messages = np.log(messages)
            # messages = self._safe_divide(messages, messages.sum(axis=1, keepdims=True))
            
            
            # check convergence 
            # calculate marginal or map
            probs = np.zeros([n_V, num_states])
            for i in range(n_V):
                probs[i] = log_b_arr[i,:]
                for j in in_neighbors[i]:
                    probs[i] += messages[index_bases[i]+in_neighbors[i].index(j)] 

            probs = self._safe_norm_exp(probs)
            
            error = (probs - old_probs)**2
            
            if len(error):
                error = np.sum(error, 1)
                error = error.mean()
                # error = np.max(np.abs(probs-old_probs))
            else:
                error = 0.
            
            # print('Iteration=%d, difference=%f\n'%(count, error))
            error_arr.append(error)
            
            bp_labels = np.zeros(n_V)
            for ii in np.arange(n_V):   
                bp_labels[ii] = np.argmax(probs[ii,:])
                
            # potential.append(np.log(compute_potential_2(bp_labels, adj, b_arr, H)))
            bp_labels[bp_labels == 0] = -1


            bp_test = bp_labels[test_idx]
            # test_acc = np.mean(gt_test_label == bp_test)
            if np.array_equal(gt_test_label, bp_test):
                test_acc = 1
            else:
                test_acc = 0
            acc_arr.append(test_acc)
            # dist.append(np.linalg.norm((gt_test_label-bp_test), ord=2))
            dist.append(np.mean(bp_test == gt_test_label))
            
            if error < epsilon: 
                converged = True
                # print(error)
                break


        # calculate marginal or map
        probs = np.zeros([n_V, num_states])
        for i in range(n_V):
            probs[i] = log_b_arr[i,:]
            for j in in_neighbors[i]:
                probs[i] += messages[index_bases[i]+in_neighbors[i].index(j)] 

        # normalize
        # results = self._safe_divide(probs, probs.sum(axis=1, keepdims=True))
        results = self._safe_norm_exp(probs)
        
        bp_labels = np.zeros(n_V)
        for ii in np.arange(n_V): 
            bp_labels[ii] = np.argmax(results[ii,:])
        # potential.append(np.log(compute_potential_2(bp_labels, adj, b_arr, H)))
        bp_labels[bp_labels == 0] = -1

                
        bp_test = bp_labels[test_idx]
        # test_acc = np.mean(gt_test_label == bp_test)
        if np.array_equal(gt_test_label, bp_test):
            test_acc = 1
        else:
            test_acc = 0
        acc_arr.append(test_acc)
        # dist.append(np.linalg.norm((gt_test_label-bp_test), ord=2))
        dist.append(np.mean(bp_test == gt_test_label))
            
        
        return results, bp_labels, acc_arr, dist, error_arr

    def run(self, graph, use_log=False, verbose=False):
        self.verbose = verbose
        
        output = self.run_one(graph[0], graph[1], graph[2], graph[3], graph[4])
        results = output[0]
        map = output[1]
        acc = output[2]
        dist = output[3]
        diff = output[4]
        
        return results, map, acc, dist, diff


if __name__ == "__main__":
    bp = BeliefPropagation_MPLP("marginal")
    
