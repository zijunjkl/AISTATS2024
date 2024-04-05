"""
Defines GGNN model based on the PGM by GNN workshop paper.
Authors: markcheu@andrew.cmu.edu, lingxiao@cmu.edu, kkorovin@cs.cmu.edu
"""

import torch
import torch.nn as nn
import time

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.normal(self.W.data, 0, 0.1) #uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.normal_(self.a.data, 0, 0.1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.relu = nn.ReLU()

    def forward(self, h_feat, adj):
        Wh = torch.mm(h_feat, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        # J = J.unsqueeze(2)
        # a_input = torch.cat((a_input, J), 2)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # masked
        # zero_vec = -9e15*torch.ones_like(e)
        # attention = torch.where(adj > 0, e, zero_vec)
        # print(attention)
        # attention = F.softmax(attention, dim=1)
        # attention = torch.sigmoid(attention)
        
        zero_vec = torch.zeros_like(e)
        attention = torch.where(adj > 0, torch.exp(e), zero_vec)
        
        
        # fully connected
        # attention = F.softmax(e, dim=1)
        
        # attention = F.dropout(attention, self.dropout, training=self.training)
        # h_prime = torch.matmul(attention, Wh)

        # if self.concat:
        #     return F.elu(h_prime)
        # else:
        #     return h_prime
        return attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class BPGNN(nn.Module):
    def __init__(self, state_dim, message_dim,hidden_unit_message_dim, hidden_unit_readout_dim, n_steps=10):
        super(BPGNN, self).__init__()

        self.state_dim = state_dim
        self.n_steps = n_steps
        self.message_dim = message_dim
        self.hidden_unit_message_dim = hidden_unit_message_dim
        self.hidden_unit_readout_dim = hidden_unit_readout_dim

        self.propagator = nn.GRUCell(self.message_dim, self.state_dim)
        self.W = nn.Linear(self.message_dim, self.state_dim)
        self.U = nn.Linear(self.message_dim, self.state_dim)
        self.message_passing = nn.Sequential(
            nn.Linear(6, self.hidden_unit_message_dim),
            # nn.Linear(2*self.state_dim+1, self.hidden_unit_message_dim),
            # 2 for each hidden state, 1 for J[i,j], 1 for b[i] and 1 for b[j]
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.message_dim),
        )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.estimate_Cij = GraphAttentionLayer(4, self.hidden_unit_message_dim, dropout=0.6, alpha=0.2, concat=True)
        self.estimate_Ci = nn.Sequential(
            nn.Linear(4, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, self.hidden_unit_message_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_unit_message_dim, 1)
            # nn.ReLU(),
            # nn.Linear(self.hidden_unit_message_dim, 1),
            # # nn.Softmax(dim=0)
            # nn.Sigmoid()
        ) 
        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)
                m.bias.data.fill_(0)

    # unbatch version for debugging
    def forward(self, J, bias, N_step, MPNNtype):
        n_nodes = len(J)
        row, col = torch.nonzero(torch.triu(J)).t()
        adj = J.clone()
        adj[torch.abs(J)>0] = 1
        bx = torch.cat((-1*bias.reshape([n_nodes, 1]),bias.reshape([n_nodes, 1])),1) #(num_node, message_dim)        
        
        row, col = torch.nonzero(J).t()
        Cij_matrix = self.estimate_Cij(torch.cat((bx, torch.sum(adj, axis =1, keepdim=True), torch.sum(J, axis =1, keepdim=True)),1), adj)
        # Cij_matrix = adj.clone()
        Cij_matrix = Cij_matrix + torch.transpose(Cij_matrix, 0, 1)
        Cij_matrix = Cij_matrix / 2
        
        Cij_matrix_reshape = Cij_matrix[row, col]#.clone().detach()
        Cij_matrix_reshape = Cij_matrix_reshape.unsqueeze(1).unsqueeze(2).repeat(1,2,2)
        
        # BP
        # Ci = 1 - torch.sum(adj, axis =1, keepdim=True)
        Ci = self.estimate_Ci(torch.cat((bx, torch.sum(adj, axis =1, keepdim=True), torch.sum(J, axis =1, keepdim=True)),1))
        
        dom_scaler = Ci + torch.sum(Cij_matrix, axis=1, keepdim=True)
        
        thres_vec = 0.1*torch.ones_like(Ci)
        dom_scaler = torch.sign(dom_scaler)*torch.where(torch.abs(dom_scaler) > 0.1, torch.abs(dom_scaler), thres_vec)
        
        # dom_scaler = torch.ones([n_nodes,1])
        
        # adj_scale = torch.mul(Cij_matrix, adj)
        adj = adj.unsqueeze(0).expand(self.message_dim, -1, -1)
        # adj_scale = adj_scale.unsqueeze(0).expand(self.message_dim, -1, -1)
        # adj_scale = torch.mul(Cij_matrix_reshape, adj)
        
        temp = bias / dom_scaler[:,0]
        bias_scale = temp#.clone().detach()
        bx_scale = torch.cat((-1*bias_scale.reshape([n_nodes, 1]),bias_scale.reshape([n_nodes, 1])),1) #(num_node, message_dim)
        
        num_edges = row.shape[0]
        Jxy = torch.zeros([num_edges, 2, 2])
        Jxy[:,0,0] = J[row, col]
        Jxy[:,0,1] = -J[row, col]
        Jxy[:,1,0] = -J[row, col]
        Jxy[:,1,1] = J[row, col]
        
        Jxy_scale = torch.zeros([num_edges, 2, 2])
        Jxy_scale = Jxy / Cij_matrix_reshape
        # J_scale = J/Cij_matrix
        # Jxy_scale[:,0,0] = J_scale[row, col]
        # Jxy_scale[:,0,1] = -J_scale[row, col]
        # Jxy_scale[:,1,0] = -J_scale[row, col]
        # Jxy_scale[:,1,1] = J_scale[row, col]
        
        # initialize edge messages e_ij = ln(mu_ij)
        edge_messages = torch.log(torch.ones(self.message_dim, n_nodes, n_nodes)) #(message_dim, num_node, num_node)
        
        # node messages n_i = sum ln(mu_ij): j:row; i:col 
        node_messages = torch.transpose(torch.sum(torch.mul(adj,edge_messages), dim=1), 0,1) #(num_node, message_dim)
        node_messages = node_messages / dom_scaler.expand(-1, self.message_dim)
        
        # normalizer z_i: a scaler
        normalizer = torch.sum(torch.mul(torch.exp(bx_scale),torch.exp(node_messages)), dim=1, keepdim = True) # (num_node, 1)
        ln_normalizer = torch.log(normalizer) # (num_node, 1)
        
        # hidden states h_i = ln(P_i)
        hidden_states = torch.cat((-ln_normalizer, -ln_normalizer), 1) + bx_scale + node_messages
        
        
        diff_arr = torch.zeros(N_step)
        
        start = time.time()
        for step in range(N_step):            
            # print(step)
            
            # calculate message form j to i: ln(mu_ji). j:row; i:col           
            ln_pj = hidden_states[row,:].unsqueeze(2).repeat(1,1,2)
            ln_zj = ln_normalizer[row,:].unsqueeze(2).repeat(1,1,2)
            
            if torch.isinf(torch.sum(ln_pj)) or torch.isnan(torch.sum(ln_pj)):
                print('error!')
                # Cij_matrix = self.estimate_Cij(bx, J, adj)
                
            ln_mu_ij = torch.transpose(edge_messages[:,col,row],0,1).unsqueeze(2).repeat(1,1,2)            
            ln_mu_ji = torch.exp(Jxy_scale + ln_pj + ln_zj - ln_mu_ij)
            
            # normalize the messages
            temp = torch.max(ln_mu_ji, dim=1)[0]
            temp2 = torch.sum(temp, dim=1, keepdim=True)
            ln_mu_ji = ln_mu_ji / temp2.unsqueeze(2).repeat(1,2,2)
            
            ln_mu_ji_scale = torch.pow(ln_mu_ji, Cij_matrix_reshape)
            ln_mu_ji_scale = torch.max(ln_mu_ji_scale, dim=1)[0]
            
            ln_mu_ji_scale = torch.log(ln_mu_ji_scale)
            edge_messages[:,row,col] = torch.transpose(ln_mu_ji_scale,0,1)
            
            # update node messages n_i = sum ln(mu_ij): j:row; i:col 
            node_messages = torch.transpose(torch.sum(torch.mul(adj,edge_messages), dim=1), 0,1) #(num_node, message_dim)
            node_messages = node_messages / dom_scaler.expand(-1, self.message_dim)
        
            # update normalizer z_i: a scaler
            normalizer = torch.sum(torch.mul(torch.exp(bx_scale),torch.exp(node_messages)), dim=1, keepdim = True) # (num_node, 1)
            ln_normalizer = torch.log(normalizer) # (num_node, 1)
            
            # update hidden states h_i = ln(P_i)
            old_hidden_states = hidden_states.clone()
            hidden_states = torch.cat((-ln_normalizer, -ln_normalizer), 1) + bx_scale + node_messages
        
            diff = torch.mean(torch.sum((hidden_states - old_hidden_states)**2, dim=1))
            
            
            if diff < 1e-7:
            # print(diff)
                break
            if torch.isinf(diff) or torch.isnan(diff):
                hidden_states = old_hidden_states.clone()
                break
            
            # print('step=%d, difference = %f'%(step, diff))
        
        computing_time = time.time()-start
        
        readout = torch.exp(hidden_states)
        # print('step=%d'%(step))
        # print(readout)
        
        ## compute pairwise marginals
        row, col = torch.nonzero(torch.triu(J)).t()
        num_edges = row.shape[0]
        J_scale = (J/Cij_matrix)#.clone().detach()
        Jxy_scale = torch.zeros([num_edges, 2, 2])
        Jxy_scale[:,0,0] = torch.exp(J_scale[row, col])
        Jxy_scale[:,0,1] = torch.exp(-J_scale[row, col])
        Jxy_scale[:,1,0] = torch.exp(-J_scale[row, col])
        Jxy_scale[:,1,1] = torch.exp(J_scale[row, col])
        
        temp_i = readout[col,:]/torch.exp(edge_messages[:, row, col].T)
        temp_i = temp_i.unsqueeze(1).repeat(1,2,1)
        
        temp_j = readout[row,:]/torch.exp(edge_messages[:, col, row].T)
        temp_j = temp_j.unsqueeze(2).repeat(1,1,2)
        
        pairwise_probs = Jxy_scale*temp_i*temp_j
        pairwise_probs = pairwise_probs/torch.sum(pairwise_probs, dim = [1,2]).unsqueeze(1).unsqueeze(2).repeat(1,2,2)
        readout_pairwise = pairwise_probs.reshape(num_edges, 4)

                            
        ## compute entropy
        Cij = Cij_matrix[row, col]
        
        temp = readout * torch.log(readout+1e-16)
        node_Hvalue = -1.0 * temp.sum(dim=1)
        
        temp = readout_pairwise * torch.log(readout_pairwise+1e-16)
        edge_Hvalue = -1.0 * temp.sum(dim=1)
        
        node_entropy = torch.matmul(Ci[:,0], node_Hvalue.clone().detach())
        edge_entropy = torch.matmul(Cij, edge_Hvalue.clone().detach())
        entropy_ = node_entropy + edge_entropy
        
        
        return readout, readout_pairwise, computing_time, entropy_, node_entropy, edge_entropy, Ci[:,0], Cij, dom_scaler
