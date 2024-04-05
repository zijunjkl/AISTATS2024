"""

Graphical model generators
Authors: kkorovin@cs.cmu.edu

"""

import numpy as np
import networkx as nx

from graphical_models.data_structs import BinaryMRF

struct_names = ["star", "random_tree", "powerlaw_tree", "path",
                "cycle", "ladder", "grid",
                "circ_ladder", "barbell", "loll", "wheel",
                "bipart", "tripart", "fc", "loop-free"]

def generate_struct_mask(struct, n_nodes, shuffle_nodes=False):
    # a horrible collection of ifs due to args in nx constructors
    if struct == "star":
        g = nx.star_graph(n_nodes)
    elif struct == "random_tree":
        g = nx.random_tree(n_nodes, seed=1) # for fixed structure
        # g = nx.random_tree(n_nodes)
    elif struct == "powerlaw_tree":
        g = nx.powerlaw_tree(n_nodes, gamma=3, seed=1)
    elif struct == "binary_tree":
        raise NotImplementedError("Implement a binary tree.")
    elif struct == "path":
        g = nx.path_graph(n_nodes)
    elif struct == "cycle":
        g = nx.cycle_graph(n_nodes)
    elif struct == "ladder":
        # g = nx.ladder_graph(5) # for 9 nodes
        g = nx.ladder_graph(8) # for 15 nodes
    elif struct == "grid":
        m = 4
        n = 4
        # m = np.random.choice(range(1, n_nodes+1))
        # n = n_nodes // m
        g = nx.grid_2d_graph(m, n)
    elif struct == "circ_ladder":
        g = nx.circular_ladder_graph(8)# for 9 nodes
        
    elif struct == "barbell":
        assert n_nodes >= 4
        # m = np.random.choice(range(2, n_nodes-1))
        # blocks = (m, n_nodes-m)
        # g = nx.barbell_graph(*(3,1))  # for 6 nodes
        # g = nx.barbell_graph(*(4,1))  # for 9 nodes
        # g = nx.barbell_graph(*(6,3))  # for 15 nodes
        g = nx.barbell_graph(*(6,4))  # for 16 nodes
    elif struct == "loll":
        assert n_nodes >= 2
        # m = np.random.choice(range(2, n_nodes+1))
        # m = 3 # 9 nodes
        m = 10 # 15 nodes
        g = nx.lollipop_graph(m, n_nodes-m)
    elif struct == "wheel":
        # g = nx.wheel_graph(n_nodes)
        g = nx.wheel_graph(n_nodes) # node 0 is the center
    elif struct == "bipart":
        # m = np.random.choice(range(1,n_nodes))
        # m = np.int(np.round(n_nodes/2))
        m = 5
        blocks = (m, n_nodes-m)
        g = nx.complete_multipartite_graph(*blocks)
    elif struct == "tripart":
        # allowed to be zero
        m, M = np.random.choice(range(1, n_nodes), size=2)
        if m > M:
            m, M = M, m
        blocks = (m, M-m, n_nodes-M)
        # g = nx.complete_multipartite_graph(*(3,3,3)) # for 9 nodes
        # g = nx.complete_multipartite_graph(*(5,5,5)) # for 15 nodes
        g = nx.complete_multipartite_graph(*(5,5,6)) # for 16 nodes
    elif struct == "fc":
        g = nx.complete_graph(n_nodes)
    else:
        raise NotImplementedError("Structure {} not implemented yet.".format(struct))

    node_order = list(range(n_nodes))
    if shuffle_nodes:
        np.random.shuffle(node_order)

    # a weird subclass by default; raises a deprecation warning
    # with a new update of networkx, this should be updated to
    # nx.convert_matrix.to_numpy_array
    # np_arr_g = nx.to_numpy_matrix(g, nodelist=node_order)
    # print(np_arr_g)
    np_arr_g = nx.to_numpy_matrix(g)
    ## for barbell with 6 nodes
    # np_arr_g = np_arr_g[0:6, 0:6] 
    # np_arr_g[2,3] = 1
    # np_arr_g[3,2] = 1
    # np_arr_g = np_arr_g[0:9, 0:9] ## for ladder with 9 nodes
    np_arr_g = np_arr_g[0:16, 0:16] ## for ladder with 9 nodes
    print(np_arr_g)
    return np_arr_g.astype(int)


def construct_binary_mrf(struct, n_nodes, shuffle_nodes=False):
    """Construct one binary MRF graphical model

    Arguments:
        struct {string} -- structure of the graph
        (on of "path", "ladder", ...)
        n_nodes {int} -- number of nodes in the graph
        shuffle_nodes {bool} -- whether to permute node labelings
                                uniformly at random
    Returns:
        BinaryMRF object
    """
    # W = np.random.normal(0., 1, (n_nodes, n_nodes))
    W = np.random.uniform(-1, 1, (n_nodes, n_nodes))
    W = np.triu(W)
    W = W + W.T
    # W = (W + W.T) / 2
    b = np.random.uniform(-0.05, 0.05, n_nodes)
    # b = np.random.normal(0., 0.25, n_nodes)
    mask = generate_struct_mask(struct, n_nodes, shuffle_nodes)
    # mask = np.zeros([5,5])
    # mask[0,1] = 1
    # mask[0,2] = 1
    # mask[0,4] = 1
    # mask[1,0] = 1
    # mask[1,3] = 1
    # mask[1,4] = 1
    # mask[2,0] = 1
    # mask[3,1] = 1
    # mask[4,0] = 1
    # mask[4,1] = 1
    
    # mask = np.zeros([9,9])
    # mask[0,1] = 1
    # mask[0,2] = 1
    # mask[0,4] = 1
    # mask[1,0] = 1
    # mask[1,3] = 1
    # mask[1,4] = 1
    # mask[2,0] = 1
    # mask[2,5] = 1
    # mask[2,7] = 1
    # mask[3,1] = 1
    # mask[4,0] = 1
    # mask[4,1] = 1
    # mask[5,2] = 1
    # mask[6,7] = 1
    # mask[6,8] = 1
    # mask[7,2] = 1
    # mask[7,6] = 1
    # mask[7,8] = 1
    # mask[8,6] = 1
    # mask[8,7] = 1
    
    # mask = np.zeros([12,12])
    # mask[0,1] = 1
    # mask[0,2] = 1
    # mask[0,4] = 1
    # mask[1,0] = 1
    # mask[1,3] = 1
    # mask[1,4] = 1
    # mask[2,0] = 1
    # mask[2,5] = 1
    # mask[2,7] = 1
    # mask[2,9] = 1
    # mask[3,1] = 1
    # mask[4,0] = 1
    # mask[4,1] = 1
    # mask[5,2] = 1
    # mask[5,10] = 1
    # mask[6,7] = 1
    # mask[6,8] = 1
    # mask[7,2] = 1
    # mask[7,6] = 1
    # mask[7,8] = 1
    # mask[8,6] = 1
    # mask[8,7] = 1
    # mask[9,2] = 1
    # mask[9,10] = 1
    # mask[9,11] = 1
    # mask[10,5] = 1
    # mask[10,9] = 1
    # mask[11,9] = 1
    mask = mask.astype(int)
    W *= mask
    return BinaryMRF(W, b, struct=struct)


if __name__ == "__main__":
    print("Testing all structures:")
    n_nodes = 5
    for struct in struct_names:
        print(struct, end=": ")
        graph = construct_binary_mrf(struct, n_nodes)
        print("ok")
        # print(graph.W, graph.b)

    print("Nodes not shuffled:")
    graph = construct_binary_mrf("wheel", n_nodes, False)
    print(graph.W, graph.b)

    print("Nodes shuffled:")
    graph = construct_binary_mrf("wheel", 5)
    print(graph.W, graph.b)

    try:
        graph = construct_binary_mrf("fully_conn", 3)
    except NotImplementedError:
        pass

