import numpy as np
import torch

###############################################
###############################################

def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############################################
###############################################

def get_layers(model, layer_class):
    """
    Recursively collect all layers of a given type from a model.

    """
    return [module for module in model.modules() if isinstance(module, layer_class)]

