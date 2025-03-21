import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MAGALoss(nn.Module):
    def __init__(self, args):
        super(MAGALoss, self).__init__()
        self.beta_kl = args.beta_kl
        self.beta_recon = args.beta_recon
        self.batch_size = args.batch_size
        

    def forward(self, x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon):
        
        # Reconstruction loss: D(E(x1, x2), x1) ≈ x2
        # sum over all elements, needed to divide by the
        recon_loss = nn.functional.binary_cross_entropy(decoded_x2, x2, reduction='none')
        recon_loss = recon_loss.sum(dim=[1,2,3])
        # KL-divergence for regularization
        mu_z = mu2 - mu1
        var_z = th.exp(logvar1) + th.exp(logvar2)
        logvar_z = th.log(var_z)
        kl_loss = -0.5 * th.sum(1+logvar_z - mu_z.pow(2) - var_z, dim=[1])

        # kl_loss = (-0.5 * th.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        #            -0.5 * th.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp()))
                
        recon_latent_loss = nn.functional.l1_loss(z_recon, z, reduction='none')
        recon_latent_loss = recon_latent_loss.sum(dim=[1])
        
        # Total loss
        total_loss = recon_loss + self.beta_kl * kl_loss + self.beta_recon * recon_latent_loss
        
        return total_loss.mean(dim=0), recon_loss.mean(dim=0), recon_latent_loss.mean(dim=0), kl_loss.mean(dim=0)
    