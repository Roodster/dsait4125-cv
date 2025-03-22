import torch as th

from src.learners._base import BaseLearner

    
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
        loss, recon_loss, kl_loss = self.criterion(decoded_x, x, mu, logvar)
        return loss, recon_loss, kl_loss

    def step(self, data_loader, results):
        self.model.train()
        train_loss = 0.0

        for x in data_loader:
            x = x.to(self.args.device)

            # Forward pass
            recon_x, mu, logvar = self.model(x)

            # Compute loss (correct argument order)
            loss, _, _ = self.compute_loss(x, mu, logvar, recon_x)

            # Backpropagation and optimization step
            self.update(loss=loss)

            # Track total loss
            train_loss += loss.item()

        # Store averaged losses in results
        num_batches = len(data_loader)
        results.train_losses = train_loss / num_batches

        return results

    def evaluate(self, data_loader, results):
        self.model.eval()
        test_loss = 0.0

        metrics = {}

        with th.no_grad():
            for i,x in enumerate(data_loader):
                
                x = x.to(self.args.device)

                recon_x, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = self.compute_loss(recon_x, mu, logvar)
                test_loss +=loss.item()

        metrics['test_losses'] = test_loss / len(data_loader)
        metrics['recon_losses'] = recon_loss / len(data_loader)
        metrics['kl_losses'] = kl_loss / len(data_loader)

        results.update(metrics)

        return results

