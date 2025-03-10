from array import array

import numpy as np
import torch

from src.args import Args
from src.networks.maga_net import MAGANet
from src.registry import setup
from src.common.utils import set_seed
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

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

def encode_img(file_path):
    data = np.load(file_path)
    images = data["imgs"][:]
    latents_values = data["latents_values"][:]
    # color, shape, scale, orientation, pos_x, pos_y
    # shape: square1, ellipse2, heart3

    ## the pivot image under different x position
    # key = np.array([[1.,3.,0.7,3.22,0.16,0.48]])
    # pivot_index = 594095
    # key = np.array([[1.,3.,0.7,3.22,0.0,0.48]])
    # pivot_index = 593935
    # key = np.array([[1.,3.,0.7,3.22,0.48,0.48]])
    # pivot_index = 594415
    ## pos = np.where(np.all(np.isclose(self.latents_values, key, atol=1e-2), axis=1))[0]
    pivot_index = 594095
    p_img = images[pivot_index]
    # color, shape, scale, orientation, pos_x, pos_y

    p_img = torch.tensor(p_img.reshape(1,1, 64, 64), dtype=torch.float32)
    imgs = torch.tensor(images.reshape(-1, 1, 64, 64), dtype=torch.float32)
    # result = torch.tensor(result.reshape(-1,1, 64, 64), dtype=torch.float32)

    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)
    model = MAGANet(args)  # Reinitialize model
    model.load_state_dict(torch.load(
        "./outputs/run_dev_maga/seed_42_010320251008/run_dev_maga/seed_42/models/model.pth"))  # Load saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    p_img = p_img.to(device)
    imgs1 = imgs[:200000].to(device)
    imgs2 = imgs[200000:400000].to(device)
    imgs3 = imgs[400000:].to(device)
    # imgs = imgs.to(device)
    model.eval()  # Set model to evaluation mode

    latent_vir = []
    for i in range(imgs.shape[0]):
        temp_i = imgs[i].reshape(1, 1, 64, 64)
        temp_i = temp_i.to(device)
        z, mu1, logvar1, mu2, logvar2 = model.encoder(p_img, temp_i)
        latent_vir.append(z.detach().cpu().numpy())

    z = np.array(latent_vir).squeeze()
    np.save("./outputs/latent_vir", z)
    print("result saved")
    return

if __name__ == "__main__":
    # encode_img('./data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    z = np.load("./outputs/latent_vir.npy")
    z_gt = np.load("./outputs/latent_vir_gt.npy")
    # color, shape, scale, orientation, pos_x, pos_y

    ### use t-sne （very slow）
    # z_embedded = TSNE(n_components=2, learning_rate='auto',
    #               init='random', perplexity=3).fit_transform(z)
    # plt.scatter(z_embedded[:, 0], z_embedded[:, 1])
    # plt.show()

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
    fig, ax = plt.subplots(1,2, figsize=(18, 10))
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
