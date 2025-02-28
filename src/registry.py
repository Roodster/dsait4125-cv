import torch as th 

from src.networks import VisionModel, MAGANet
from src.learners import ClassificationLearner, MAGALearner
from src.losses import MAGALoss

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
    }

}


def setup(algo_id):
    """ Get the model, algorithm and buffer classes for the given algorithm."""
    algo_id = algo_id.lower()

    if algo_id not in REGISTRY:
        raise ValueError(f"Algorithm {algo_id} not found in registry.")
    
    return REGISTRY[algo_id]
