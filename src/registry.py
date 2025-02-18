import torch as th 

from src.networks import *
from src.learners.learner import *


REGISTRY = {
    'base': {
        'model': VisionModel,
        'learner': ClassificationLearner,
        'optimizer': th.optim.Adam,
        'criterion': th.nn.CrossEntropyLoss
    }

}


def setup(algo_id):
    """ Get the model, algorithm and buffer classes for the given algorithm."""
    algo_id = algo_id.lower()

    if algo_id not in REGISTRY:
        raise ValueError(f"Algorithm {algo_id} not found in registry.")
    
    return REGISTRY[algo_id]
