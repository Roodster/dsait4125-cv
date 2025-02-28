import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from src.args import Args
from src.registry import setup
from src.dataset import DspritesDataset, get_dataloaders_2element, BinarySyntheticDataset, get_dataloaders
from src.experiment import Experiment
from src.common.utils import set_seed
<<<<<<< HEAD
from src.networks.MAGANet import MAGANet, kl_divergence, latent_reconstruction_loss
=======
>>>>>>> fb2a905 (Updates)

def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    train_data = DspritesDataset("./data/2d/train.npz")
    test_data = DspritesDataset("./data/2d/test.npz")
    train_loader, test_loader = get_dataloaders_2element(train_data, test_data,
                                                batch_size=args.batch_size)

    # dataset = BinarySyntheticDataset(num_classes=3, n_samples_per_class=100, img_size=64, pattern_type='geometric', seed=42, noise_prob=0.05)
    # train_loader, test_loader, val_loader = get_dataloaders(dataset=dataset, train_ratio=0.7, test_ratio=0.2, batch_size=32)

    # initialize experiment
    experiment = Experiment(registry=registry, 
                            args=args 
                            )
    
    # run experiment
    experiment.run(train_loader=train_loader, test_loader=test_loader)
    

<<<<<<< HEAD
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
=======
>>>>>>> fb2a905 (Updates)

if __name__ == "__main__":
    main()