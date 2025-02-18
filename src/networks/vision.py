import torch as th
import torch.nn as nn

from src.networks.modules.common._base import BaseModel
from src.networks.modules.cnn import CNN

class VisionModel(BaseModel):

    def __init__(self, args):
        super(VisionModel, self).__init__(args=args)

        self.network = CNN(in_channels=3, num_classes=2)
        
    def forward(self, x):
        y_pred = self.network(x)
        return y_pred


