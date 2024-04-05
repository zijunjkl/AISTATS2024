"""

Graphical model class
Authors: kkorovin@cs.cmu.edu

TODO:
* MST generation in BinaryMRF
"""

import networkx as nx
import numpy as np
from inference import get_algorithm

dflt_algo = {"marginal": "bp", "map": "bp"}


class GraphicalModel:
    def __init__(self, n_nodes, params=None, default_algo=dflt_algo):
        """Constructor

        Arguments:
            n_nodes {int} - number of vertices in graphical model
            params {dictionary<str,np.array> or None} -- parameters of the model

        Keyword Arguments:
            default_algo {dict} -- default inference methods to use,
            unless they are overriden in corresponding methods
            (default: {dflt_algo})
        """
        self.algo_marginal = default_algo["marginal"]
        self.algo_map = default_algo["map"]

    def set_ground_truth(self, unary_marginal_est=None, pairwise_marginal_est=None, map_est=None, unary_map_onehot=None, pairwise_map_onehot=None):
        """ Setting labels:
        To be used when instantiating
        a model from saved parameters
        """
        self.unary_marginal = unary_marginal_est
        self.pairwise_marginal = pairwise_marginal_est
        self.map = map_est
        self.unary_map_onehot = unary_map_onehot
        self.pairwise_map_onehot = pairwise_map_onehot

    # Running inference with/without Inference object
    def get_marginals(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_marginal
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="marginal")
        return inf_res

    def get_map(self, algo_obj=None, algo=None):
        if algo_obj is None:
            if algo is None:
                algo = self.algo_map
            algo_obj = get_algorithm(algo)
        inf_res = algo_obj.run(self, mode="map")
        return inf_res

    def __repr__(self):
        return "GraphicalModel:{} on {} nodes".format(
            self.__class__.__name__, self.n_nodes)


class BinaryMRF(GraphicalModel):
    def __init__(self, W, b, struct=None):
        """Constructor of BinaryMRF class.

        Arguments:
            W {np.array} -- (N, N) matrix of pairwise parameters
            b {np.array} -- (N,) vector of unary parameters
        
        Keyword Arguments:
            struct {string or None} -- description of graph structure
                                       (default: {None})
        """
        self.W = W
        self.b = b
        self.struct = struct
        self.n_nodes = len(W)
        self.default_algo = {"marginal": "bp",
                             "map": "bp"}
        # params = {"W": W, "b": b}
        super(BinaryMRF, self).__init__(
            n_nodes=self.n_nodes,
            default_algo=self.default_algo)

    def get_subgraph_on_nodes(self, node_list):
        """ node_list does not need to be ordered,
            return in the same order
        """
        nx_graph = nx.from_numpy_matrix(self.W)
        sg = nx_graph.subgraph(node_list)
        W_sg = np.array(nx.to_numpy_matrix(sg))
        b_sg = self.b[node_list]  # in the same order
        return BinaryMRF(W_sg, b_sg)

    def get_max_abs_spanning_tree(self):
        nx_graph = nx.from_numpy_matrix(np.abs(self.W))
        tree = nx.minimum_spanning_tree(nx_graph)
        W_abs_tree = np.array(nx.to_numpy_matrix(tree))
        W_mask = np.where(W_abs_tree > 0, 1, 0)
        # zero out unused edges:
        W_tree = W_mask * self.W
        b_tree = self.b
        return BinaryMRF(W_tree, b_tree)


if __name__ == "__main__":
    print(get_algorithm("bp"))
