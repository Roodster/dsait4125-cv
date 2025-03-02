import numpy as np
import torch

from src.args import Args
from src.networks.maga_net import MAGANet
from src.registry import setup
from src.common.utils import set_seed
import matplotlib.pyplot as plt

def load_img(file_path):
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
    values = latents_values
    # color, shape, scale, orientation, pos_x, pos_y
    shape_ = values[:, 1]
    scale = values[:, 2]
    rotation = values[:, 3]
    position_x = values[:, 4]
    position_y = values[:, 5]

    # Mask only change pos_y
    # shape: square1, ellipse2, heart3
    # mask = (
    #         (shape_ == 3.) &
    #         (scale == 0.7) &
    #         (rotation == 3.22) &
    #         (position_x == 0.16)
    # )

    # Mask only change pos_x
    mask = (
            (shape_ == 3.) &
            (scale == 0.7) &
            (np.isclose(rotation, 3.22214631, atol=1e-2)) &
            (np.isclose(position_y, 0.48387097, atol=1e-2))
    )
    result = images[mask]
    p_img = torch.tensor(p_img.reshape(1,1, 64, 64), dtype=torch.float32)
    result = torch.tensor(result.reshape(-1,1, 64, 64), dtype=torch.float32)
    return p_img, result

if __name__ == "__main__":
    p_img, imgs = load_img("./data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    # print(imgs.shape)

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
    imgs = imgs.to(device)
    model.eval()  # Set model to evaluation mode

    latent_vir = []
    for i in range(imgs.shape[0]):
        temp_i = imgs[i].reshape(1, 1, 64, 64)
        z, mu1, logvar1, mu2, logvar2 = model.encoder(p_img,temp_i)
        latent_vir.append(z.detach().cpu().numpy())

    plt.imshow(np.array(latent_vir).squeeze())
    plt.show()