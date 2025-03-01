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

from src.networks.maga_net import MAGANet



if __name__ == "__main__":
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    # dataset = SyntheticDataset(num_classes=2, n_samples_per_class=128, x_dim=3, y_dim=64, z_dim=64)
    train_data = DspritesDataset("./data/2d/train.npz")
    test_data = DspritesDataset("./data/2d/test.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                         batch_size=args.batch_size)


    # Training Loop
    latent_dim = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MAGANet(args).cuda()  # Move to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer
    loss_fn = nn.BCELoss()

    num_epochs = 3

    model.train()
    print("Start Training")
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, (x1, x2) in enumerate(train_loader):
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

    torch.save(model.state_dict(), "./outputs/magan_model.pth")