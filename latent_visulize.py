from array import array

import numpy as np
import torch
from sklearn.utils import shuffle

from src.args import Args
from src.networks import VAE
from src.networks.maga_net import MAGANet
from src.registry import setup
from src.common.utils import set_seed
from src.dataset import DspritesDataset, get_dataloaders_2element

import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.decomposition import PCA

def get_mask(file_path, mask_num):
    # color, shape, scale, orientation, pos_x, pos_y
    data = np.load(file_path)
    latents_values = data["latents_values"][:]

    # shape: square1, ellipse2, heart3

    ## the pivot image under different x position
    # key = np.array([[1.,3.,0.7,3.22,0.16,0.48]])
    # pivot_index = 594095
    # key = np.array([[1.,3.,0.7,3.22,0.0,0.48]])
    # pivot_index = 593935
    # key = np.array([[1.,3.,0.7,3.22,0.48,0.48]])
    # pivot_index = 594415
    ## pos = np.where(np.all(np.isclose(self.latents_values, key, atol=1e-2), axis=1))[0]

    # color, shape, scale, orientation, pos_x, pos_y
    # shape: square1, ellipse2, heart3
    shape_ = latents_values[:, 1]
    scale = latents_values[:, 2]
    rotation = latents_values[:, 3]
    position_x = latents_values[:, 4]
    position_y = latents_values[:, 5]

    # Mask only change pos_y
    if mask_num == 0:
        # Mask only change pos_x
        mask = (
                (shape_ == 3.) &
                (scale == 0.7) &
                (np.isclose(rotation, 3.22214631, atol=1e-2)) &
                (np.isclose(position_y, 0.48387097, atol=1e-2))
        )
    if mask_num == 1:
        # mask for orientation
        mask = (
                (shape_ == 3.) &
                (scale == 0.7) &
                # (np.isclose(position_x, 0.16, atol=1e-2)) &
                (np.isclose(position_y, 0.48387097, atol=1e-2))
        )
    if mask_num == 2:
        mask = (
            (shape_ == 3.) &
            # (scale == 0.7) &
            (np.isclose(position_x, 0.16, atol=1e-2))
            # (np.isclose(position_y, 0.48387097, atol=1e-2))
        )

    # mask for size
    if mask_num == 4: # use only x
        mask = (
                # (shape_ == 3.) &
                # (scale == 0.7) &
                (np.isclose(rotation, 3.22214631, atol=1e-2)) &
                # (np.isclose(position_x, 0.16, atol=1e-2)) &
                (np.isclose(position_y, 0.48387097, atol=1e-2))
        )
    if mask_num == 5: # use only y
        mask = (
                # (shape_ == 3.) &
                # (scale == 0.7) &
                (np.isclose(rotation, 3.22214631, atol=1e-2)) &
                (np.isclose(position_x, 0.16, atol=1e-2))
                # (np.isclose(position_y, 0.48387097, atol=1e-2))
        )
    result = mask
    return result

def encode_img(train_loader, test_loader, model, args):
    model.eval()  # Set model to evaluation mode

    latent_vir = []

    single_output = args.model_name == "vae"
    if single_output:
        for batch_idx, x1 in enumerate(test_loader):
            x1 = x1.to(args.device)  # Move tensors to GPU if available
            x, mu, logvar = model(x1)  # Forward pass
            z = model.reparameterize(mu, logvar)
            latent_vir.append(z.detach().cpu().numpy())

        for batch_idx, x1 in enumerate(train_loader):
            x1 = x1.to(args.device)  # Move tensors to GPU if available
            x, mu, logvar = model(x1)  # Forward pass
            z = model.reparameterize(mu, logvar)
            latent_vir.append(z.detach().cpu().numpy())
    else:
        for batch_idx, (x1, x2) in enumerate(test_loader):
            x1, x2 = x1.to(args.device), x2.to(args.device)  # Move tensors to GPU if available
            z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2  = model(x1, x2)  # Forward pass
            latent_vir.append(z.detach().cpu().numpy())

        for batch_idx, (x1, x2) in enumerate(train_loader):
            x1, x2 = x1.to(args.device), x2.to(args.device)  # Move tensors to GPU if available
            z, mu1, logvar1, mu2, logvar2, decoded_x1, decoded_x2  = model(x1, x2)  # Forward pass
            latent_vir.append(z.detach().cpu().numpy())

    z = np.vstack(latent_vir)
    np.save("./outputs/latent_vir", z)
    print("result saved")
    return

def generate_files(path):
    args = Args(file=f"{path}/hyperparameters.yaml")

    single_output = args.model_name == "vae"
    train_data = DspritesDataset("./data/2d/train.npz", single_output=single_output)
    test_data = DspritesDataset("./data/2d/test.npz", single_output=single_output)

    z_gt = np.concatenate([test_data.latents_values,train_data.latents_values])
    np.save("./outputs/latent_vir_gt", z_gt)
    print("ground truth latent value saved")

    train_loader, test_loader = get_dataloaders_2element(
        train_data, test_data,
        batch_size=args.batch_size,
        shuffle=False
    )

    if args.model_name == "maga":
        model = MAGANet(args)
        model.load_state_dict(torch.load(
            f"{path}/models/model_2element.pth"))
        model = model.to(args.device)
    elif args.model_name == "vae":
        model = VAE(args)
        model.load_state_dict(torch.load(
            f"{path}/models/model_2element.pth"))
        model = model.to(args.device)


    encode_img(train_loader, test_loader, model, args)


if __name__ == "__main__":
    generate_files("./outputs/run_dev_maga/seed_2_250320250829")

    z = np.load("./outputs/latent_vir.npy")
    # z = np.load("./outputs/latent_vir_2range.npy")
    z_gt = np.load("./outputs/latent_vir_gt.npy")
    # color, shape, scale, orientation, pos_x, pos_y

    ### use t-sne （very slow）
    # z_embedded = TSNE(n_components=2, learning_rate='auto',
    #               init='random', perplexity=3).fit_transform(z)
    # plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
    # plt.show()

    ### use PCA (can't cope non-linear)
    ## see shape
    # classes = ['square', 'ellipse', 'heart']
    # z = z[0:-1:10]
    # z_gt1 = z_gt[0:-1:10][:, 1].reshape(-1)  # shape
    # pca = PCA(n_components=2)
    # z_embedded = pca.fit_transform(z)
    # fig, ax = plt.subplots(1, figsize=(10, 7))
    # plt.scatter(*z_embedded.T, s=0.3, c=z_gt1 - 1, alpha=0.3)
    # plt.setp(ax, xticks=[], yticks=[])
    # cbar = plt.colorbar(boundaries=np.arange(4) - 0.5, spacing='uniform')
    # cbar.set_ticks(np.array([0, 1, 2]))
    # cbar.set_ticklabels(classes)
    # plt.title("Color by shape")
    # plt.show()
    # # plt.savefig("./outputs/shape.jpg", dpi=100, bbox_inches='tight')
    # plt.close(fig)

    ### use umap
    ## see shape
    classes = ['square', 'ellipse', 'heart']
    z = z[0:-1:10]
    z_gt1 = z_gt[0:-1:10][:, 1].reshape(-1)  # shape
    z_embedded = umap.UMAP(n_neighbors=50).fit_transform(z)
    fig, ax = plt.subplots(1, figsize=(10, 7))
    plt.scatter(*z_embedded.T, s=0.3, c=z_gt1-1, alpha=0.3)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar(boundaries=np.arange(4) - 0.5, spacing='uniform')
    cbar.set_ticks(np.array([0,1,2]))
    cbar.set_ticklabels(classes)
    plt.title("Color by shape")
    plt.show()
    # plt.savefig("./outputs/shape.jpg", dpi=100, bbox_inches='tight')
    plt.close(fig)

    ## see position
    fig, ax = plt.subplots(1,2, figsize=(16, 7))
    z_gt1 = z_gt[0:-1:10][:, -2].reshape(-1, 1)  # pos_x
    sc1 = ax[0].scatter(*z_embedded.T, s=0.3, c=z_gt1, alpha=0.3)
    ax[0].set_title("Color by pos_x")

    z_gt2 = z_gt[0:-1:10][:, -1].reshape(-1)  # Ensure proper slicing
    sc2 = ax[1].scatter(*z_embedded.T, s=0.3, c=z_gt2, alpha=0.3)
    ax[1].set_title("Color by pos_y")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(sc1, ax=ax[0])
    fig.colorbar(sc2, ax=ax[1])
    # plt.title("Color by position")
    plt.show()
    # plt.savefig("./outputs/position.jpg", dpi=100, bbox_inches='tight')
    plt.close(fig)

    ## see scale
    mask = get_mask("./data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 4)
    z1 = np.load("./outputs/latent_vir.npy")
    z1 = z1[mask]
    z_gt1 = z_gt[mask][:, 2].reshape(-1)  # scale
    z_embedded1 = umap.UMAP(n_neighbors=50).fit_transform(z1)

    mask = get_mask("./data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 5)
    z2 = np.load("./outputs/latent_vir.npy")
    z2 = z2[mask]
    z_gt2 = z_gt[mask][:, 2].reshape(-1)  # scale
    z_embedded2 = umap.UMAP(n_neighbors=50).fit_transform(z1)

    fig, ax = plt.subplots(1,2, figsize=(18, 10))
    sc1 = ax[0].scatter(*z_embedded1.T, s=10, c=z_gt1, alpha=0.5)
    ax[0].set_title("Color by scale (fixing pos_y)")
    sc2 = ax[1].scatter(*z_embedded2.T, s=10, c=z_gt2, alpha=0.5)
    ax[1].set_title("Color by scale (fixing pos_x)")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(sc1, ax=ax[0])
    fig.colorbar(sc2, ax=ax[1])
    plt.show()
    plt.close(fig)

    ## see orientation
    # mask = get_mask("./data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 1)
    # z = np.load("./outputs/latent_vir.npy")
    # z = z[mask]#[0:-1:5]
    # z_gt1 = z_gt[mask][:, 3].reshape(-1)  # orientation
    # z_embedded = umap.UMAP(n_neighbors=50,n_components=2).fit_transform(z)
    # fig, ax = plt.subplots(1, figsize=(14, 10))
    # plt.scatter(*z_embedded.T, s=20, c=z_gt1, cmap='hsv', alpha=0.3) # change to a cyclic cmap
    # plt.setp(ax, xticks=[], yticks=[])
    # cbar = plt.colorbar()
    # cbar.set_ticks(z_gt1)
    # plt.show()
