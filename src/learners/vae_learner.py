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
        loss, loss_recon, loss_kl = self.criterion(decoded_x, x, mu, logvar)
        return loss, loss_recon, loss_kl

    def step(self, data_loader, results):
        self.model.train()
        train_loss = 0.0
        train_loss_kl = .0
        train_loss_recon = .0

        for x in data_loader:
            x = x.to(self.args.device)

            # Forward pass
            recon_x, mu, logvar = self.model(x)

            # Compute loss (correct argument order)
            loss, loss_recon, loss_kl = self.compute_loss(x, mu, logvar, recon_x)

            # Backpropagation and optimization step
            self.update(loss=loss)

            # Track total loss
            train_loss += loss.item()
            train_loss_kl += loss_kl.item()
            train_loss_recon += loss_recon.item()

        # Store averaged losses in results
        N = len(data_loader)
        results.train_losses = train_loss / N
        results.train_recon_losses = train_loss_recon / N
        results.train_kl_losses = train_loss_kl / N

        return results

    def evaluate(self, data_loader, results):
        self.model.eval()
        test_loss = 0.0
        test_loss_kl = 0.0
        test_loss_recon = 0.0

        metrics = {}

        with th.no_grad():
            for i,x in enumerate(data_loader):
                
                x = x.to(self.args.device)

                recon_x, mu, logvar = self.model(x)
                loss, recon_loss, kl_loss = self.compute_loss(x, mu, logvar, recon_x)
                test_loss += loss.item()
                test_loss_kl += kl_loss.item()
                test_loss_recon += recon_loss.item()

        N = len(data_loader)
        metrics['test_losses'] = test_loss / N
        metrics['test_kl_losses'] = test_loss_kl / N
        metrics['test_recon_losses'] = test_loss_recon / N
        results.generated_images = recon_x.cpu().detach().numpy().squeeze()
        results.update(metrics)

        return results

