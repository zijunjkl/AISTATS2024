"""

Interface for inference algorithms
Authors: kkorovin@cs.cmu.edu

"""

from inference.bp_homophilic import BeliefPropagation
from inference.bp_TreeReweighted import BeliefPropagation_TRW
from inference.bp_mplp import BeliefPropagation_MPLP
from inference.exact import ExactInference
from inference.mcmc import GibbsSampling
from inference.gnn_inference_bp_general_form_learnable import MPNNInference_General_GT
from inference.gnn_inference import GatedGNNInference


def get_algorithm(algo_name):
    """ Returns a constructor """
    if algo_name == "bp":
        return BeliefPropagation
    elif algo_name == "TRWbp":
        return BeliefPropagation_TRW
    elif algo_name == "MPLP":
        return BeliefPropagation_MPLP
    elif algo_name == "exact":
        return ExactInference
    elif algo_name == "mcmc":
        return GibbsSampling
    elif algo_name == "gnn_inference":
        return GatedGNNInference
    ###
    elif algo_name == 'mpnn_general_form_gt':
        return MPNNInference_General_GT
    else:
        raise ValueError("Inference algorithm {} not supported".format(algo_name))