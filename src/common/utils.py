import yaml
import os
import glob
import random as rnd
import numpy as np
import torch as th



def load_config(config_file=None):
    assert config_file is not None, "Error: config file not found."
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args(args_file):
    config = load_config(config_file=args_file)
    return config

def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.Generator().manual_seed(seed)
    
    
def freeze_params(model):
    
    for param in model.parameters():
        param.required_grad = False
        
    return model
    