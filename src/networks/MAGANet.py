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
    def __init__(self, latent_dim=10):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TransformationModule(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, z1, z2):
        delta = torch.cat([z1, z2], dim=1)
        return self.fc(delta)

class Decoder(nn.Module):
    def __init__(self, latent_dim=10, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 8, 8)
        return self.deconv(x)


class MAGANet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.transform = TransformationModule(latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z_trans = self.transform(z1, z2)
        x_transformed = self.decoder(z_trans)
        return x_transformed


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

    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    print("Start Training")
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (x1, x2) in enumerate(train_loader):  # Assuming `dataloader` is defined
            # x1np = x1.numpy()
            x1, x2 = x1.to(device), x2.to(device)  # Move tensors to GPU if available

            optimizer.zero_grad()  # Zero gradients

            x_transformed = model(x1, x2)  # Forward pass

            loss = loss_fn(x_transformed, x2)  # Compute BCE loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
