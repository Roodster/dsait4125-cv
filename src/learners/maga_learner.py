import torch as th

from src.learners._base import BaseLearner

    
class MAGALearner(BaseLearner):

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
    
    def compute_loss(self, x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon):
        
        loss =  self.criterion(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)
        return loss
    
    
    def step(self, data_loader, results):
        self.model.train()
        train_loss = .0
        
        for x1, x2 in data_loader:
            
            x1, x2 = x1.to(self.args.device), x2.to(self.args.device)

            z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2 = self.model(x1, x2)      
            z_recon = self.model.compute_z_reconstruction(x1, decoded_x1)
            loss, recon_loss, recon_latent_loss, kl_loss = self.compute_loss(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)    
            
            self.update(loss=loss)

            train_loss += loss.item()

        results.train_losses = train_loss / len(data_loader)
        return results
    
    
    def evaluate(self, data_loader, results):
        self.model.eval()
        test_loss = 0.0

        metrics = {}

        with th.no_grad():
            for i, (x1, x2) in enumerate(data_loader):
                
                x1, x2 = x1.to(self.args.device), x2.to(self.args.device)
                                
                z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2 = self.model(x1, x2)      
                z_recon, _, _, _, _, _ , _ = self.model(x1, decoded_x1)
                loss, recon_loss, recon_latent_loss, kl_loss = self.compute_loss(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)                        
                test_loss += loss.item()

        metrics['test_losses'] = test_loss / len(data_loader)
        metrics['recon_losses'] = recon_loss / len(data_loader)
        metrics['recon_latent_losses'] = recon_latent_loss / len(data_loader)
        metrics['kl_losses'] = kl_loss / len(data_loader)

        results.update(metrics)

        return results
