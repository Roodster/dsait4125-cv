import torch as th
import torch.nn as nn

from src.networks.modules.common._base import BaseModel
        
class InferenceModel(BaseModel):

    def __init__(self, args, layers):
        super(InferenceModel, self).__init__(args=args)
        
        if not isinstance(layers, list):
            layers = [layers]
            
        self.layers = layers
        
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)
        
        return x


