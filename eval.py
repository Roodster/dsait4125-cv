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
    train_data = DspritesDataset("./data/2d/train.npz")
    test_data = DspritesDataset("./data/2d/test.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                batch_size=args.batch_size)


    model = MAGANet(args)  # Reinitialize model
    # model.load_state_dict(torch.load(
    #   "./outputs/run_dev_maga/seed_42_100320251508/run_dev_maga/seed_42/models/model2range.pth"))  # 2range except square
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_070320251453/run_dev_maga/seed_42/models/model2range.pth"))  # 2range except ellipsis
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_010320251008/run_dev_maga/seed_42/models/model.pth"))  # 2element except ellipsis
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_170320250845/run_dev_maga/seed_42/models/model_2element.pth"))  # 2element k = 10
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_170320251701/run_dev_maga/seed_42/models/model_2element.pth"))  # 2element k = 0.1
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_170320252142/run_dev_maga/seed_42/models/model_2element.pth"))  # 2element k = 2
    # model.load_state_dict(torch.load(
    #     "./outputs/run_dev_maga/seed_42_170320252217/run_dev_maga/seed_42/models/model_2element.pth"))  # 2element k = 0
    model.load_state_dict(torch.load(
        "./outputs/run_dev_maga/seed_42_190320250901/run_dev_maga/seed_42/models/model_2element.pth"))  # 2element k = 3
    model = model.to(args.device)
    model.eval()  # Set model to evaluation mode

    loss_fn = MAGALoss(args)
    running_loss = 0.0
    print("start testing...")
    generated_image = None
    x1_sample = None  # Store x1 for visualization
    x2_sample = None  # Store x2 for visualization
    for batch_idx, (x1, x2) in enumerate(test_loader):
        x1, x2 = x1.to(args.device), x2.to(args.device)  # Move tensors to GPU if available

        z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2  = model(x1, x2)  # Forward pass
        z_recon = model.compute_z_reconstruction(x1, decoded_x1)
        loss,_,_,_ = loss_fn(x2, z, mu1, logvar1, mu2, logvar2, decoded_x2, z_recon)

        running_loss += loss.item()

        if batch_idx == 18:
            generated_image = decoded_x1.cpu().detach().numpy().squeeze()
            x1_sample = x1.cpu().detach().numpy().squeeze()
            x2_sample = x2.cpu().detach().numpy().squeeze()
    avg_loss = running_loss / len(test_loader)
    print(f"Test average loss: {avg_loss}")

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

def eval_compare_element_range():
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    # dataset = SyntheticDataset(num_classes=2, n_samples_per_class=128, x_dim=3, y_dim=64, z_dim=64)
    train_data = DspritesDataset("./data/2d/train2range.npz")
    test_data = DspritesDataset("./data/2d/test.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                         batch_size=args.batch_size)
    model_element = MAGANet(args)
    model_range = MAGANet(args)
    model_range.load_state_dict(torch.load(
        "./outputs/run_dev_maga/seed_42_070320251453/run_dev_maga/seed_42/models/model2range.pth"))  # 2range except ellipsis
    model_element.load_state_dict(torch.load(
        "./outputs/run_dev_maga/seed_42_010320251008/run_dev_maga/seed_42/models/model.pth"))  # 2element except ellipsis

    model_element = model_element.to(args.device)
    model_range = model_range.to(args.device)
    model_element.eval()  # Set model to evaluation mode
    model_range.eval()

    loss_fn = MAGALoss(args)
    running_loss = 0.0
    print("start testing...")
    generated_image = None
    x1_sample = None  # Store x1 for visualization
    x2_sample = None  # Store x2 for visualization

    x1, x2 = next(iter(test_loader))
    x1, x2 = x1.to(args.device), x2.to(args.device)  # Move tensors to GPU if available


    z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2 = model_element(x1, x2)  # Forward pass
    z_recon = model_element.compute_z_reconstruction(x1, decoded_x1)
    generated_image = decoded_x1.cpu().detach().numpy().squeeze()[:10]
    x2_sample = x2.cpu().detach().numpy().squeeze()[:10]

    z_range, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2 = model_range(x1, x2)  # Forward pass
    z_recon = model_element.compute_z_reconstruction(x1, decoded_x1)
    generated_image_range = decoded_x1.cpu().detach().numpy().squeeze()[:10]

    images_plot = np.concatenate([x2_sample,generated_image,generated_image_range])
    fig, axes = plt.subplots(3, 10, figsize=(16, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images_plot[i], cmap="gray")  # Change cmap if images are colored
        ax.axis("off")  # Hide axis for better visualization

    row_labels = ["GT", "to element", "to range"]
    # Add labels to the left of each row
    for row, label in enumerate(row_labels):
        fig.text(0.1, 0.82 - (row * 0.33), label, va='center', ha='right', fontsize=12, fontweight="bold")
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    # eval_compare_element_range()