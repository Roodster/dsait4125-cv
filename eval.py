import torch
from torch.utils.data import DataLoader

import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

from src.args import Args
from src.registry import setup
from src.dataset import get_dataloaders,  DspritesDataset, get_dataloaders_2element
from src.experiment import Experiment
from src.common.utils import set_seed
from src.networks.maga_net import MAGANet, kl_divergence, latent_reconstruction_loss
from src.losses import MAGALoss

def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    # dataset = SyntheticDataset(num_classes=2, n_samples_per_class=128, x_dim=3, y_dim=64, z_dim=64)
    train_data = DspritesDataset("./data/2d/train2range.npz")
    test_data = DspritesDataset("./data/2d/test2range.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                batch_size=args.batch_size)


    model = MAGANet(args)  # Reinitialize model
    model.load_state_dict(torch.load("./outputs/run_dev_maga/seed_42_100320251508/run_dev_maga/seed_42/models/model2range.pth"))  # Load saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    loss_fn = MAGALoss(args)
    running_loss = 0.0
    print("start testing...")
    generated_image = None
    x1_sample = None  # Store x1 for visualization
    x2_sample = None  # Store x2 for visualization
    beta_kl = 1
    beta_lr = 1
    for batch_idx, (x1, x2) in enumerate(test_loader):
        x1, x2 = x1.to(device), x2.to(device)  # Move tensors to GPU if available

        z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2  = model(x1, x2)  # Forward pass
        z_recon = model.compute_z_reconstruction(x1, decoded_x1)
        loss,_,_,_ = loss_fn(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)

        running_loss += loss.item()

        if batch_idx == 1:
            generated_image = decoded_x1.cpu().detach().numpy().squeeze()
            x1_sample = x1.cpu().detach().numpy().squeeze()
            x2_sample = x2.cpu().detach().numpy().squeeze()
    avg_loss = running_loss / len(test_loader) /args.batch_size
    print(f"Test average loss: {avg_loss}")
    # Convert tensor to NumPy for visualization

    # Define grid size (8x8 = 64 images)
    num_images = 64  # Choose how many images to display
    grid_size = int(np.sqrt(num_images))  # 8x8 grid

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_image[i], cmap="gray")  # Change cmap if images are colored
        ax.axis("off")  # Hide axis for better visualization

    plt.tight_layout()
    plt.show()

    # Display the generated image
    # if generated_image is not None:
    #     fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    #
    #     # Ensure grayscale images are displayed correctly
    #     axes[0].imshow(x1_sample[0], cmap='gray')  # x1 sample
    #     axes[0].set_title("Input Image (x1)-pivot")
    #     axes[0].axis("off")
    #
    #     axes[1].imshow(x2_sample[480], cmap='gray')  # x2 sample (ground truth)
    #     axes[1].set_title("Ground Truth (x2)-varying")
    #     axes[1].axis("off")
    #
    #     axes[2].imshow(generated_image[480], cmap='gray')  # Generated x2
    #     axes[2].set_title("Generated Image")
    #     axes[2].axis("off")
    #
    #     plt.show()

if __name__ == "__main__":
    main()