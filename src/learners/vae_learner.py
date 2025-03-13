import torch as th

from ._base import BaseLearner

    
class VAELearner(BaseLearner):
    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None
                 ):
        super().__init__(args=args, model=model, optimizer=optimizer, criterion=criterion)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def compute_loss(self, x, mu, logvar, decoded_x):
        loss = self.criterion(x, mu, logvar, decoded_x)
        return loss

    def step(self, data_loader, results):
        pass
    
    
    def evaluate(self, data_loader, results):
        pass