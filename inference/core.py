"""
Base Inference class: inference engines that can function in marginal and MAP regimes.
All inference classes (BP, GNN etc) should subclass this class.
Authors: kkorovin@cs.cmu.edu

"""

class Inference:
    def __init__(self, mode):
        if mode not in ["marginal", "map"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.mode = mode

    def reset_mode(self, mode):
        if mode not in ["marginal", "map"]:
            raise ValueError("Inference mode {} not supported".format(mode))
        self.mode = mode

        
    def run(self, graphs, verbose=False):
        raise NotImplementedError("Implement in a child class.")

    def __repr__(self):
        return "Algorithm {} in mode {}".format(self.__class__.__name__, self.mode)
