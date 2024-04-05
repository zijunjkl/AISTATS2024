"""

Runnable experiments module
Authors: kkorovin@cs.cmu.edu

TODO:
* make cmd argument parser to choose which exp to run

NOTE:
* m[0] = p(-1), m[1] = p(+1)
"""

import os
import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

from inference import get_algorithm
from experiments.exp_helpers import get_dataset_by_name
from constants import *


# Train-test pairs ------------------------------------------------------------

def in_sample_experiment(struct):
    train_set_name = struct + "_small"
    test_set_name  = struct + "_small"
    run_experiment(train_set_name, test_set_name)

def out_of_sample_experiment(struct,struct2):
    """ Test generalization to same- and different- structure
        larger graphs """
    train_set_name = struct #+ "_small"
    test_set_name = struct2 #+ "_medium" # "conn_medium", "trees_medium"
    run_experiment(train_set_name, test_set_name)

def upscaling_experiment(struct):
    """ trainset here combines a few structures,
        testset is increasingly large 
    """
    train_set_name = struct + "_small"
    test_set_name  = struct + "_large"
    run_experiment(train_set_name, test_set_name)

def in_sample_experiment_map(struct):
    train_set_name = struct + "_small"
    test_set_name  = struct + "_small"
    run_experiment(train_set_name, test_set_name, "map")

# Large-scale experiments -----------------------------------------------------
def approx_trees_experiment():
    train_set_name = "trees_approx"
    test_set_name = "trees_approx"
    run_experiment(train_set_name, test_set_name, "marginal")

def approx_nontrees_experiment():
    train_set_name = "nontrees_approx"
    test_set_name = "nontrees_approx"
    run_experiment(train_set_name, test_set_name, "marginal")

def approx_barbell_experiment():
    train_set_name = "barbell_approx"
    test_set_name = "barbell_approx"
    run_experiment(train_set_name, test_set_name, "marginal")

def approx_fc_experiment():
    train_set_name = "fc_approx"
    test_set_name = "fc_approx"
    run_experiment(train_set_name, test_set_name, "marginal")

# Runner ----------------------------------------------------------------------

def run_experiment(train_set_name, test_set_name, inference_mode="marginal",
                   base_data_dir=DFLT_DATA_DIR, model_base_dir=DFLT_MODEL_DIR):
    """
    tests for in-sample (same structure, same size, marginals)
    """
    train_path = os.path.join(base_data_dir, "train")
    test_path = os.path.join(base_data_dir, "test")
    model_load_path = os.path.join(model_base_dir, train_set_name)

    train_data = get_dataset_by_name(train_set_name, train_path)
    test_data  = get_dataset_by_name(test_set_name, test_path, mode=inference_mode)
 
    # load model
    n_hidden_states = 5
    message_dim_P = 5
    hidden_unit_message_dim = 64 
    hidden_unit_readout_dim = 64
    T = 10
    gnn_constructor = get_algorithm("gnn_inference")
    gnn_inference = gnn_constructor(inference_mode, n_hidden_states, 
                                    message_dim_P,hidden_unit_message_dim,
                                    hidden_unit_readout_dim, T,
                                    model_load_path, USE_SPARSE_GNN)

    # run inference on test
    times = {}

    t0 = time()
    gnn_res = gnn_inference.run(test_data, DEVICE)
    times["gnn"] = (time()-t0) / len(test_data)
    
    t0 = time()
    bp = get_algorithm("bp")(inference_mode)
    bp_res = bp.run(test_data, use_log=True, verbose=False)
    times["bp"] = (time()-t0) / len(test_data)

    # TODO! don't forget to uncomment
    t0 = time()
    mcmc = get_algorithm("mcmc")(inference_mode)
    mcmc_res = mcmc.run(test_data)
    times["mcmc"] = (time()-t0) / len(test_data)

    #--- sanity check ----#
    #exact = get_algorithm("exact")("marginal")
    #exact_res = exact.run(test_data)
    #--- sanity check ----#

    # all loaded graphs have ground truth set
    if inference_mode == "marginal":
        true_labels = []
        for g in test_data:
            true_labels.extend(list(m[1] for m in g.marginal))

        gnn_labels = []
        for graph_res in gnn_res:
            gnn_labels.extend(list(m[1] for m in graph_res))

        bp_labels = []
        for graph_res in bp_res:
            bp_labels.extend(list(m[1] for m in graph_res))

        #mcmc_labels = bp_labels  # TODO! don't forget to uncomment
        mcmc_labels = []
        for graph_res in mcmc_res:
            mcmc_labels.extend(list(m[1] for m in graph_res))

        #--- sanity check ----#
        # exact_labels = []
        # for graph_res in exact_res:
        #     exact_labels.extend(list(m[1] for m in graph_res))
        #--- sanity check ----#

        # colors = []
        # for g in test_data:
        #     colors.extend([g.struct] * g.n_nodes)

        # save these results
        save_marginal_results(true_labels, gnn_labels, bp_labels, mcmc_labels,
            filename="./experiments/saved_exp_res/res_{}_{}".format(train_set_name, test_set_name))

        # plot them
        plot_marginal_results_individual(true_labels, gnn_labels, bp_labels, mcmc_labels,
            filename="./experiments/res_{}_{}".format(train_set_name, test_set_name))

    # MAP: only numeric
    else:
        true_labels = []
        for g in test_data:
            true_labels.extend(g.map)
        true_labels = np.array(true_labels)

        gnn_labels = []
        for graph_res in gnn_res:
            gnn_labels.extend(list(-1 if m[0]>m[1] else +1 for m in graph_res))
        gnn_labels = np.array(gnn_labels)
        gnn_accuracy = np.mean(true_labels == gnn_labels)

        bp_labels = []
        for graph_res in bp_res:
            bp_labels.extend(graph_res)
        bp_labels = np.array(bp_labels)
        bp_accuracy = np.mean(true_labels == bp_labels)

        mcmc_labels = []
        for graph_res in mcmc_res:
            mcmc_labels.extend(graph_res)
        mcmc_labels = np.array(mcmc_labels)
        mcmc_accuracy = np.mean(true_labels == mcmc_labels)

        print("Accuracies: GNN {}, BP {}, MCMC {}".format(gnn_accuracy, bp_accuracy, mcmc_accuracy))
    print("Runtimes", times)

def parse_exp_args():
    parser = argparse.ArgumentParser()
    # critical arguments, change them
    parser.add_argument('--exp_name', type=str,
                        help='name of experiment to run')
    args = parser.parse_args()
    return args

def save_marginal_results(true_labels, gnn_labels, bp_labels, mcmc_labels, filename, colors=None):
    res = {'true_labels': true_labels, 'gnn_labels': gnn_labels, 'bp_labels': bp_labels,
            'mcmc_labels': mcmc_labels, 'colors': colors}
    np.save(filename, res, allow_pickle=True)

def plot_marginal_results_individual(true_labels, gnn_labels, bp_labels, mcmc_labels, filename):
    fsize=(10,10)
    col = 'purple'

    def plot_one(true, algo, prefix):
        fig = plt.figure(figsize=fsize)
        ax=fig.add_subplot(1,1,1)
        plt.axis('off')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        diag = np.linspace(0,1,num=200)
        plt.scatter(true, algo, s=7, c=col, alpha=0.3)
        plt.plot(diag,diag,c='red',linewidth=3, alpha=0.7)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig(filename+prefix, bbox_inches=extent, pad_inches=0)
        plt.clf()

    plot_one(true_labels, gnn_labels, '_gnn')
    plot_one(true_labels, bp_labels, '_bp')
    plot_one(true_labels, mcmc_labels, '_mcmc')


def plot_marginal_results(true_labels, gnn_labels, bp_labels, mcmc_labels, filename):
    plt.title("Inference results")
    plt.axis('off')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
    ax1.set_title("GNN", fontsize=40)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.scatter(true_labels, gnn_labels)

    ax2.set_title("BP", fontsize=40)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.scatter(true_labels, bp_labels)

    ax3.set_title("MCMC", fontsize=40)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.scatter(true_labels, mcmc_labels)

    #--- sanity check ----#
    #ax4.set_title("Exact (just a sanity check)")
    #ax4.scatter(true_labels, exact_labels)
    #--- sanity check ----#

    plt.savefig(filename)

def plot_marginal_results_with_colors(true_labels, gnn_labels, bp_labels, mcmc_labels, colors, filename):
    cols = ['red', 'green', 'blue', 'purple']
    map_to_col = {s: cols[i] for i,s in enumerate(list(set(colors)))}

    plt.title("Inference results")
    plt.axis('off')
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30, 10))
    ax1.set_title("GNN", fontsize=40)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    for c in set(colors):
        # find labels that should be plotted
        true_ = [l for i, l in enumerate(true_labels) if colors[i] == c]
        algo_ = [l for i, l in enumerate(gnn_labels) if colors[i] == c]
        ax1.scatter(true_, algo_, c=map_to_col[c], label=c, alpha=0.3)
    ax1.legend()

    ax2.set_title("BP", fontsize=40)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    # ax2.scatter(true_labels, bp_labels)
    for c in set(colors):
        # find labels that should be plotted
        true_ = [l for i, l in enumerate(true_labels) if colors[i] == c]
        algo_ = [l for i, l in enumerate(bp_labels) if colors[i] == c]
        ax2.scatter(true_, algo_, c=map_to_col[c], label=c, alpha=0.3)
    ax2.legend()

    ax3.set_title("MCMC", fontsize=40)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    for c in set(colors):
        # find labels that should be plotted
        true_ = [l for i, l in enumerate(true_labels) if colors[i] == c]
        algo_ = [l for i, l in enumerate(mcmc_labels) if colors[i] == c]
        ax3.scatter(true_, algo_, c=map_to_col[c], label=c, alpha=0.3)
    ax3.legend()

    #--- sanity check ----#
    #ax4.set_title("Exact (just a sanity check)")
    #ax4.scatter(true_labels, exact_labels)
    #--- sanity check ----#

    plt.savefig(filename)


if __name__ == "__main__":
    args = parse_exp_args()
    if args.exp_name.startswith("in_sample"):
        struct = args.exp_name[len("in_sample_"):]
        in_sample_experiment(struct=struct)
    elif args.exp_name == "out_sample":
        out_of_sample_experiment("fc_small","fc_small")
    elif args.exp_name == "upscaling":
        upscaling_experiment("barbell")
    elif args.exp_name == "in_sample_map":
        in_sample_experiment_map(struct="fc")
    elif args.exp_name == "trees_approx":
        approx_trees_experiment()
    elif args.exp_name == "nontrees_approx":
        approx_nontrees_experiment()
    elif args.exp_name == "barbell_approx":
        approx_barbell_experiment()
    elif args.exp_name == "fc_approx":
        approx_fc_experiment()
    elif args.exp_name.startswith('res'):
        path = f"./experiments/saved_exp_res/{args.exp_name}.npy"
        filename = f"./experiments/{args.exp_name}"
        data = np.load(path, allow_pickle=True)[()]
        true_labels = data['true_labels']
        gnn_labels = data['gnn_labels']
        bp_labels = data['bp_labels']
        mcmc_labels = data['mcmc_labels']
        # colors = data['colors']
        plot_marginal_results_individual(true_labels, gnn_labels, bp_labels, mcmc_labels, filename)
    else:
        raise ValueError(f"Unrecognized experiment `{args.exp_name}`")



