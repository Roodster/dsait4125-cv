import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

from src.args import Args
from src.registry import setup
from src.dataset import get_dataloaders, SyntheticDataset, DspritesDataset, get_dataloaders_2element
from src.experiment import Experiment
from src.common.utils import set_seed
from src.networks.maga_net import MAGANet, kl_divergence, latent_reconstruction_loss

def main():
        
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


    model = MAGANet(in_channels=1, latent_dim=10)  # Reinitialize model
    model.load_state_dict(torch.load("./outputs/magan_model.pth"))  # Load saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    loss_fn = nn.BCELoss()
    running_loss = 0.0
    print("start testing...")
    generated_image = None
    x1_sample = None  # Store x1 for visualization
    x2_sample = None  # Store x2 for visualization
    beta_kl = 1
    beta_lr = 1
    for batch_idx, (x1, x2) in enumerate(test_loader):
        x1, x2 = x1.to(device), x2.to(device)  # Move tensors to GPU if available

        x_transformed, mu1, logvar1, mu2, logvar2  = model(x1, x2)  # Forward pass

        loss_bce = loss_fn(x_transformed, x2)  # Compute BCE loss
        loss_kl = kl_divergence(mu1, logvar1) + kl_divergence(mu2, logvar2)  # KL loss

        loss_recon_latent = latent_reconstruction_loss(model.encoder, model.decoder, x1,
                                                       mu1)  # latent_reconstruction_loss

        # Final loss function
        loss = loss_bce + beta_kl * loss_kl + beta_lr * loss_recon_latent

        running_loss += loss.item()

        if batch_idx == 3:
            generated_image = x_transformed.cpu().detach().numpy().squeeze()
            x1_sample = x1.cpu().detach().numpy().squeeze()
            x2_sample = x2.cpu().detach().numpy().squeeze()
    avg_loss = running_loss / len(test_loader)
    print(f"Test average loss: {avg_loss}")
    # Convert tensor to NumPy for visualization

    # Display the generated image
    if generated_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))

        # Ensure grayscale images are displayed correctly
        axes[0].imshow(x1_sample[0], cmap='gray')  # x1 sample
        axes[0].set_title("Input Image (x1)-pivot")
        axes[0].axis("off")

        axes[1].imshow(x2_sample[0], cmap='gray')  # x2 sample (ground truth)
        axes[1].set_title("Ground Truth (x2)-varying")
        axes[1].axis("off")

        axes[2].imshow(generated_image[0], cmap='gray')  # Generated x2
        axes[2].set_title("Generated Image")
        axes[2].axis("off")

        plt.show()

if __name__ == "__main__":
    main()