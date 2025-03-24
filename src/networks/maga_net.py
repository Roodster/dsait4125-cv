import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, latent_dim=10, in_channels=1):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*latent_dim)
        )

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log-variance to standard deviation
        eps = torch.randn_like(std)  # Sample from N(0, I)
        return mu + eps * std

    def forward(self, x1, x2):
        h1 = self.conv(x1)
        h2 = self.conv(x2)

        h1 = h1.view(h1.size(0), -1)
        h2 = h2.view(h2.size(0), -1)

        h1 = self.fc(h1)
        h2 = self.fc(h2)

        mu1, logvar1 = torch.chunk(h1, chunks=2, dim=-1)
        mu2, logvar2 = torch.chunk(h2, chunks=2, dim=-1)

        z1 = self.sample_z(mu1,logvar1)
        z2 = self.sample_z(mu2,logvar2)
        return z2-z1, mu1, logvar1, mu2, logvar2


# Encoder Network
class AblationEncoder(nn.Module):
    def __init__(self, latent_dim=10, in_channels=1):
        super(AblationEncoder, self).__init__()
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

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log-variance to standard deviation
        eps = torch.randn_like(std)  # Sample from N(0, I)
        return mu + eps * std

    def forward(self, x1, x2):
        h1 = self.conv(x1)  # Extract features
        mu1 = self.fc_mu(h1)  # Get mean
        logvar1 = self.fc_logvar(h1)  # Get log-variance

        h2 = self.conv(x2)  # Extract features
        mu2 = self.fc_mu(h2)  # Get mean
        logvar2 = self.fc_logvar(h2)  # Get log-variance

        z1 = self.sample_z(mu1,logvar1)
        z2 = self.sample_z(mu2,logvar2)

        return z2-z1, mu1, logvar1, mu2, logvar2


class ActNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        return x * torch.exp(self.log_scale) + self.bias

class Invertible1x1Conv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(in_channels).unsqueeze(2).unsqueeze(3))  # Identity matrix

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight)

class AdditiveCoupling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # Split channels
        x2 = x2 + self.net(x1)  # Add transformation
        return torch.cat([x1, x2], dim=1)

class FlowStep(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.act_norm = ActNorm(in_channels)
        self.inv_conv = Invertible1x1Conv(in_channels)
        self.coupling = AdditiveCoupling(in_channels)

    def forward(self, x):
        x = self.act_norm(x)
        x = self.inv_conv(x)
        x = self.coupling(x)
        return x

class SqueezeLayer(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        return x

class UnsqueezeLayer(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape

        # Reshape channels to separate spatial factors
        x = x.view(B, C // 4, 2, 2, H, W)

        # Permute dimensions to correct order
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()

        # Reshape to expanded spatial size
        x = x.view(B, C // 4, H * 2, W * 2)
        return x


class FlowModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.squeeze = SqueezeLayer()
        self.flow_steps = nn.Sequential(
            FlowStep(in_channels * 4),
            FlowStep(in_channels * 4),
            FlowStep(in_channels * 4),
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = self.flow_steps(x)
        return x


class Affine(nn.Module):
    """Affine applying x' + Mz transformation."""
    def __init__(self, latent_dim, in_channels, height, width):
        super().__init__()
        self.fc_M = nn.Linear(latent_dim, in_channels * height * width)

    def forward(self, x, z):
        """ Apply affine transformation: x' + Mz """
        B, C, H, W = x.shape
        Mz = self.fc_M(z).view(B, C, H, W)
        return x + Mz


class FlowNet(nn.Module):
    """ Decoder that learns the group action α: Z × X → X """

    def __init__(self, in_channels, latent_dim, height=64, width=64):
        super().__init__()
        self.affine = Affine(latent_dim, in_channels,height,width)  # Modulate x1 using z

        self.flow_modules = nn.Sequential(
            FlowModule(in_channels),
            FlowModule(in_channels * 4),
            FlowModule(in_channels * 16),
        )
        self.unsqueeze = nn.Sequential(
            UnsqueezeLayer(),
            UnsqueezeLayer(),
            UnsqueezeLayer(),
        )

    def forward(self, z, x1):
        """ z: latent vector (B, latent_dim), x1: input image (B, 1, 64, 64) """
        x1_transformed = self.affine(x1, z)  # Apply FiLM transformation
        x2 = self.flow_modules(x1_transformed)  # Apply flow-based transformations
        x2 = self.unsqueeze(x2)
        return torch.sigmoid(x2) # ensure intensity [0,1]


class MAGANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(latent_dim=args.latent_dim, in_channels=args.in_channels)
        self.decoder = FlowNet(in_channels=args.in_channels, latent_dim=args.latent_dim)

    def forward(self, x1, x2):
        z, mu1, logvar1, mu2, logvar2 = self.encoder(x1, x2)
        decoded_x1 = self.decoder(z, x1)
        decoded_x2 = self.decoder(z, x1)  # Decoder generates x2 using z and x1
        return z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2

    def compute_z_reconstruction(self, x1, decoded_x2):
        """Compute the reconstructed z given x1 and the decoded x1."""
        z_recon, mu1, logvar1, mu_rec, logvar_rec = self.encoder(x1, decoded_x2)
        return z_recon


class AblationMAGANet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = AblationEncoder(latent_dim=args.latent_dim, in_channels=args.in_channels)
        self.decoder = FlowNet(in_channels=args.in_channels, latent_dim=args.latent_dim)

    def forward(self, x1, x2):
        z, mu1, logvar1, mu2, logvar2 = self.encoder(x1, x2)
        decoded_x1 = self.decoder(z, x1)
        decoded_x2 = self.decoder(z, x1)  # Decoder generates x2 using z and x1
        return z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2

    def compute_z_reconstruction(self, x1, decoded_x2):
        """Compute the reconstructed z given x1 and the decoded x1."""
        z_recon, mu1, logvar1, mu_rec, logvar_rec = self.encoder(x1, decoded_x2)
        return z_recon


def kl_divergence(mu, logvar):
    """ Compute KL divergence loss """
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

def latent_reconstruction_loss(encoder, decoder, x, z):
    """ Compute L_recon_latent = || E(x, D(z, x)) - z ||_1 """
    x_transformed = decoder(z, x)  # Apply transformation D(z, x)
    z_reconstructed, _, _, _, _ = encoder(x, x_transformed)  # Get E(x, D(z, x))
    return torch.mean(torch.abs(z_reconstructed - z))  # L1 norm

if __name__ == "__main__":
    from types import SimpleNamespace
    args = SimpleNamespace(in_channels=1, latent_dim=10)
    model = MAGANet(args)
    x1 = torch.randn(1, 1, 64, 64)
    x2 = torch.randn(1, 1, 64, 64)
    print(model(x1, x2))