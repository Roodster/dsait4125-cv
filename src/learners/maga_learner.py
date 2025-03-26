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
        train_loss_recon = .0
        train_loss_kl = .0
        train_loss_recon_latent = .0
        
        for x1, x2 in data_loader:
            
            x1, x2 = x1.to(self.args.device), x2.to(self.args.device)

            z, mu1, logvar1, mu2, logvar2, decoded_x2 = self.model(x1, x2)      
            z_recon = self.model.compute_z_reconstruction(x1, decoded_x2)
            loss, recon_loss, recon_latent_loss, kl_loss = self.compute_loss(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)    
            print(f"loss: {loss} recon: {recon_loss} recon_latent: {recon_latent_loss} kl_loss: {kl_loss}")
            
            self.update(loss=loss)

            print("loss: ", loss)
            print("loss item: ", loss.item())

            train_loss += loss.item()
            train_loss_recon += recon_loss.item()
            train_loss_kl += kl_loss.item()
            train_loss_recon_latent += recon_latent_loss.item()

        N = len(data_loader)
        results.train_losses = train_loss / N
        results.train_recon_losses = train_loss_recon / N
        results.train_kl_losses = train_loss_kl / N
        results.train_recon_latent_losses = train_loss_recon_latent / N

        return results
    
    
    def evaluate(self, data_loader, results):
        self.model.eval()
        test_loss = 0.0
        test_loss_kl = 0.0
        test_loss_recon = 0.0
        test_loss_recon_latent = 0.0

        metrics = {}

        with th.no_grad():
            for i, (x1, x2) in enumerate(data_loader):
                
                x1, x2 = x1.to(self.args.device), x2.to(self.args.device)
                                
                z, mu1, logvar1, mu2, logvar2, decoded_x2 = self.model(x1, x2)      
                z_recon, _, _, _, _, _ = self.model(x1, decoded_x2)
                loss, recon_loss, recon_latent_loss, kl_loss = self.compute_loss(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)                        
                test_loss += loss.item()
                test_loss_kl += kl_loss.item()
                test_loss_recon += recon_loss.item()
                test_loss_recon_latent += recon_latent_loss.item()

        N = len(data_loader)
        metrics['test_losses'] = test_loss / N
        metrics['test_recon_losses'] = test_loss_recon / N
        metrics['test_recon_latent_losses'] = test_loss_recon_latent / N
        metrics['test_kl_losses'] = test_loss_kl / N
        results.update(metrics)
        # metrics['generated_images'] = decoded_x1.cpu().detach().numpy()
        results.generated_images = decoded_x2.cpu().detach().numpy().squeeze()
        return results
