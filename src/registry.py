import torch as th 

from networks import VisionModel, MAGANet, VAE
from learners import ClassificationLearner, MAGALearner, VAELearner
from losses import MAGALoss, VAELoss

REGISTRY = {
    'base': {
        'model': VisionModel,
        'learner': ClassificationLearner,
        'optimizer': th.optim.Adam,
        'criterion': th.nn.CrossEntropyLoss
    },
    'maga': {
        'model': MAGANet,
        'learner': MAGALearner,
        'optimizer': th.optim.Adam,
        'criterion': MAGALoss
    },
    'vae': {
        'model': VAE,
        'learner': VAELearner, #need to create this class,
        'optimizer': th.optim.Adam,
        'criterion': VAELoss
    }

}


def setup(algo_id):
    """ Get the model, algorithm and buffer classes for the given algorithm."""
    algo_id = algo_id.lower()

    if algo_id not in REGISTRY:
        raise ValueError(f"Algorithm {algo_id} not found in registry.")
    
    return REGISTRY[algo_id]
