import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta_kl=1.0):
        super(VAELoss, self).__init__()
        self.beta_kl = beta_kl  # KL-divergence weight

    def forward(self, recon_x, x, mu, logvar):

        # Reconstruction loss (Binary Cross-Entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL Divergence loss (Regularization term)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss (weighted KL divergence)
        total_loss = recon_loss + self.beta_kl * kl_loss

        return total_loss.mean(), recon_loss.mean(), kl_loss.mean()