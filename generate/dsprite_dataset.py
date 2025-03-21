import pathlib
import numpy as np
import os
from PIL import Image

def prepare_2d_data(file_path):
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
    # pivot_image = np.tile(p_img, (images.shape[0], 1, 1))

    total_size = len(images)
    values = latents_values
    # color, shape, scale, orientation, pos_x, pos_y
    shape_ = values[:, 1]
    scale = values[:, 2]
    rotation = values[:, 3]
    position_x = values[:, 4]
    position_y = values[:, 5]

    # Define the mask for exclusion
    # shape: square1, ellipse2, heart3
    mask = (
            (shape_ == 2.) &
            (position_x >= 0.6) &
            (position_y >= 0.6) &
            (rotation >= 2.0943) & (rotation <= 4.1888) &
            (scale < 0.6)
    )

    ## Save the split datasets
    os.makedirs(root, exist_ok=True)
    np.savez(root / "train.npz", imgs=images[~mask], pivot_image=p_img, latents_values=values[~mask])
    np.savez(root / "test.npz", imgs=images[mask], pivot_image=p_img, latents_values=values[mask])
    # np.savez(root / "train.npz", imgs=images[~mask],pivot_image=pivot_image[~mask], latents_values=values[~mask])
    # np.savez(root / "test.npz", imgs=images[mask], pivot_image=pivot_image[mask], latents_values=values[mask])
    print("saved files")

def prepare_2d_data_2range(file_path):
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
    pivot_image = np.tile(p_img, (images.shape[0], 1, 1))

    total_size = len(images)
    values = latents_values
    # color, shape, scale, orientation, pos_x, pos_y
    shape_ = values[:, 1]
    scale = values[:, 2]
    rotation = values[:, 3]
    position_x = values[:, 4]
    position_y = values[:, 5]

    # Define the mask for exclusion
    # shape: square1, ellipse2, heart3
    mask = (
            (shape_ == 1.) &
            (position_x > 0.5)
    )

    ## Save the split datasets
    os.makedirs(root, exist_ok=True)
    np.savez(root / "train2range.npz", imgs=images[~mask], pivot_image=pivot_image[~mask], latents_values=values[~mask])
    np.savez(root / "test2range.npz", imgs=images[mask], pivot_image=pivot_image[mask], latents_values=values[mask])
    print("saved files")


if __name__ == "__main__":
    root = pathlib.Path(__file__).parent.parent / "data" / "2d"
    prepare_2d_data(root / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")