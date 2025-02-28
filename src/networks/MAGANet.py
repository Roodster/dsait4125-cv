import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from src.args import Args
from src.registry import setup
from src.dataset import DspritesDataset, get_dataloaders_2element
from src.common.utils import set_seed

import matplotlib.pyplot as plt

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, latent_dim=10, input_channels=1):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
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

class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        return x * torch.exp(self.log_scale) + self.bias

class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(num_channels).unsqueeze(2).unsqueeze(3))  # Identity matrix

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight)

class AdditiveCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels // 2, num_channels // 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # Split channels
        x2 = x2 + self.net(x1)  # Add transformation
        return torch.cat([x1, x2], dim=1)

class FlowStep(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.act_norm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1Conv(num_channels)
        self.coupling = AdditiveCoupling(num_channels)

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
    def __init__(self, num_channels):
        super().__init__()
        self.squeeze = SqueezeLayer()
        self.flow_steps = nn.Sequential(
            FlowStep(num_channels * 4),
            FlowStep(num_channels * 4),
            FlowStep(num_channels * 4),
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = self.flow_steps(x)
        return x


class Affine(nn.Module):
    """Affine applying x' + Mz transformation."""
    def __init__(self, latent_dim, num_channels, height, width):
        super().__init__()
        self.fc_M = nn.Linear(latent_dim, num_channels * height * width)

    def forward(self, x, z):
        """ Apply affine transformation: x' + Mz """
        B, C, H, W = x.shape
        Mz = self.fc_M(z).view(B, C, H, W)
        return x + Mz


class FlowNet(nn.Module):
    """ Decoder that learns the group action α: Z × X → X """

    def __init__(self, num_channels, latent_dim, height=64, width=64):
        super().__init__()
        self.affine = Affine(latent_dim, num_channels,height,width)  # Modulate x1 using z

        self.flow_modules = nn.Sequential(
            FlowModule(num_channels),
            FlowModule(num_channels * 4),
            FlowModule(num_channels * 16),
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
    def __init__(self, in_channels=1, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(latent_dim,in_channels)
        self.decoder = FlowNet(in_channels, latent_dim)

    def forward(self, x1, x2):
        z, mu1, logvar1, mu2, logvar2 = self.encoder(x1, x2)
        generated_x2 = self.decoder(z, x1)  # Decoder generates x2 using z and x1
        return generated_x2, mu1, logvar1, mu2, logvar2

def kl_divergence(mu, logvar):
    """ Compute KL divergence loss """
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

def latent_reconstruction_loss(encoder, decoder, x, z):
    """ Compute L_recon_latent = || E(x, D(z, x)) - z ||_1 """
    x_transformed = decoder(z, x)  # Apply transformation D(z, x)
    z_reconstructed, _, _, _, _ = encoder(x, x_transformed)  # Get E(x, D(z, x))
    return torch.mean(torch.abs(z_reconstructed - z))  # L1 norm

if __name__ == "__main__":
    # Load arguments
    args = Args(file="../../data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    # dataset = SyntheticDataset(num_classes=2, n_samples_per_class=128, x_dim=3, y_dim=64, z_dim=64)
    train_data = DspritesDataset("../../data/2d/train.npz")
    test_data = DspritesDataset("../../data/2d/test.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                         batch_size=args.batch_size)


    # Training Loop
    latent_dim = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAGANet(in_channels=1, latent_dim=10).cuda()  # Move to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer
    loss_fn = nn.BCELoss()

    num_epochs = 100
    beta_kl = 300
    beta_lr = 300

    model.train()
    print("Start Training")
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (x1, x2) in enumerate(train_loader):
            # x1np = x1.numpy()
            x1, x2 = x1.to(device), x2.to(device)  # Move tensors to GPU if available

            optimizer.zero_grad()  # Zero gradients

            x_transformed, mu1, logvar1, mu2, logvar2 = model(x1, x2)  # Forward pass

            loss_bce = loss_fn(x_transformed, x2)  # Compute BCE loss
            loss_kl = kl_divergence(mu1, logvar1) + kl_divergence(mu2, logvar2)  # KL loss

            loss_recon_latent = latent_reconstruction_loss(model.encoder, model.decoder, x1, mu1)  # latent_reconstruction_loss

            # Final loss function
            loss = loss_bce + beta_kl * loss_kl + beta_lr * loss_recon_latent
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "../../outputs/magan_model.pth")