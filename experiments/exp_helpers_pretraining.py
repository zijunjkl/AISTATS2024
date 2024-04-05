"""

Experiment specifications:
an experiment is defined by train,test dataset pair,
each dataset is loaded from graphical_models/datasets.
Authors: kkorovin@cs.cmu.edu

"""

import os
import numpy as np

from graphical_models import BinaryMRF
from inference import get_algorithm
from graphical_models.data_gen import struct_names
from constants import *


# Give specs in form structure->size
# when used for train, the same is model name
data_specs = {
    "debug": 
            {"star": [5],
              "fc":   []},
    "larger_debug": 
            {"star": [10],
              "fc":   []},
}

# add simple datasets
data_specs.update({struct+"_small": {struct: [6]} for struct in struct_names})
assert "star_small" in data_specs

# add compound datasets
data_specs.update({struct+"_medium": {struct: [15,16,17]} for struct in struct_names})
data_specs.update({"trees_medium": {"star": [15, 16, 17],
                                    "path": [15, 16, 17],
                                    },
                    "conn_medium": {"bipart": [15, 16, 17],
                                    # "tripart": [15, 16, 17],
                                    "fc": [15, 16, 17],
                                    },
                  })
data_specs.update({"path_large": {"path":  [15,16,17]},
                  "fc_large": {"fc": [15,16,17]},
                  "barbell_large": {"barbell": [15,16,17]},
                  "ladder_large": {"ladder": [15,16,17]},
                  "random_tree_large": {"random_tree": [15,16,17]},
                  "wheel_large": {"wheel": [15,16,17]},
                 })


# Add experiments for part 2: Trees+BP
data_specs.update({"trees_approx": {"random_tree":  [100]},
                 })

# Add experiments for part 2: NonTrees+MCMC
data_specs.update({"nontrees_approx": 
                        {"barbell":  [100],
                        "fc":  [100]},
                    "barbell_approx": 
                        {"barbell":  [100]},
                    "fc_approx": 
                        {"fc":  [100]}
                 })

# Data loading ----------------------------------------------------------------
def get_dataset_by_name(specs_name, data_dir, mode=None):
    """
    Assumes graphs live as
    graphical_models/datasets/{train/val/test}  <-- data_dir
                                    |-- star/
                                    |    |-  9/<file1.npy>, <file2.npy> ...
                                    |    |- 10/
                                         |- 11/
                                   ...  ...
    Loads all graphs of given size and structure,
    this needs to be updated in the future
    (so that we can train and test on the same structures)

    Arguments:
        specs_name - key to the data_specs dictionary
        data_dir - train or test directory
        mode - map or marginal
    """
    if specs_name not in data_specs:
        raise ValueError("Specification {} not supported".format(specs_name))
    specs = data_specs[specs_name]
    graphs = []
    for struct in specs:
        size_list = specs[struct]
        for size in size_list:
            # go to specified dir, load and append
            directory = os.path.join(data_dir, struct, str(size))

            for filename in os.listdir(directory):
                if filename.endswith(".npy"):
                    path_to_graph = os.path.join(directory, filename)
                    data_dict = np.load(path_to_graph, allow_pickle=True)[()]  # funny indexing
                    graph = BinaryMRF(data_dict["W"], data_dict["b"])
                    graph.set_ground_truth(marginal_est=data_dict["marginal"],
                                           map_est=data_dict["map"])
                    graph.struct = struct
                    graph.optimal_mu = data_dict["optimal_mu"] ##### added !!!!!!!!
                    graphs.append(graph)

    if mode is not None:
        graphs = [g for g in graphs if getattr(g, mode) is not None]
    print("Loaded {} graphs".format(len(graphs)))
    return graphs


# Some simple checks ----------------------------------------------------------
if __name__ == "__main__":
    train_data = get_dataset_by_name("debug")
    print(train_data[0])
    print("W, b:", train_data[0].W, train_data[0].b)
    print("Marginals:", train_data[0].marginal)
    print("MAP:", train_data[0].map)

