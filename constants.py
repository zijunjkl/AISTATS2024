"""

Convenient constants

"""

import torch

DFLT_DATA_DIR = "./graphical_models/datasets/"
DFLT_MODEL_DIR = "./inference/pretrained"
USE_SPARSE_GNN = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')