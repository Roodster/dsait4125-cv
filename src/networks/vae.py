import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten()  # Flatten before passing to FC layers
        )

        self.fc_mu = nn.Linear(4 * 4 * 256, latent_dim)  # Mean of latent distribution
        self.fc_logvar = nn.Linear(4 * 4 * 256, latent_dim)  # Log-variance

    def forward(self, x):
        x = self.conv(x)  # Extract features
        mu = self.fc_mu(x)  # Get mean
        logvar = self.fc_logvar(x)  # Get log-variance
        return mu, logvar  # Return both for sampling


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Normalize output between 0 and 1
        )

    def forward(self, x):
        return self.deconv(x)


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dim=args.latent_dim)

        self.decoder_fc = nn.Linear(args.latent_dim, 4 * 4 * 256)  # Map latent dim to 4x4x256 feature space
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick"""
        std = torch.exp(0.5 * logvar)  # Convert log-variance to std dev
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std  # Reparameterization trick

    def forward(self, x):
        mu, logvar = self.encoder(x)  # Encode input to latent space
        z = self.reparameterize(mu, logvar)  # Sample latent vector
        x = self.decoder_fc(z).view(-1, 256, 4, 4)  # Reshape FC output into feature map
        x = self.decoder(x)  # Decode back to original space
        return x, mu, logvar  # Return reconstructed image and latent parameters


if __name__ == "__main__":
    pass
