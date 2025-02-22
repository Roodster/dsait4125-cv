import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SyntheticDataset(Dataset):
    def __init__(self, num_classes, n_samples_per_class, x_dim, y_dim=None, z_dim=None, labels_dim=(1,), seed=42, is_non_linear=True, noise_std=0.1):
        self.num_classes = num_classes
        self.n_samples_per_class = n_samples_per_class
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.labels_dim = labels_dim  # Tuple defining label dimensions
        self.seed = seed
        self.noise_std = noise_std
        self.is_non_linear = is_non_linear
        
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.dimensions = self._get_dimensions()
        self.data, self.labels = self._generate_data()

    def _get_dimensions(self):
        dims = [self.x_dim]
        if self.y_dim is not None:
            dims.append(self.y_dim)
        if self.z_dim is not None:
            dims.append(self.z_dim)
        return tuple(dims)

    def _apply_nonlinear_transform(self, x, class_idx):
        # Example non-linear transformation (you can customize this)
        return torch.sin(x + class_idx) + x**2 * 0.1

    def _generate_data(self):
        data = []
        labels = []

        for class_idx in range(self.num_classes):
            # Generate initial random data
            class_samples = torch.randn(self.n_samples_per_class, *self.dimensions)
            
            # Apply non-linear transformation
            if self.is_non_linear: 
                class_samples = self._apply_nonlinear_transform(class_samples, class_idx)
            else: 
                offset = class_idx * torch.ones(*self.dimensions) # Corrected offset application
                class_samples += offset
                
            data.append(class_samples)

            # Generate multi-dimensional labels
            class_labels = torch.randint(0, self.num_classes, (self.n_samples_per_class, *self.labels_dim)).squeeze(-1) # Use labels_dim
            labels.append(class_labels.long())

        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)  # Concatenate multi-dimensional labels

        return data, labels

    def _apply_nonlinear_transform(self, samples, class_idx):
        # Apply sine wave transformation
        freq = 1 + class_idx * 0.5  # Different frequency for each class
        amplitude = 2 + class_idx * 0.5  # Different amplitude for each class
        
        # Apply transformation to first dimension
        samples[:, 0] = amplitude * torch.sin(freq * samples[:, 0])
        
        # If more than one dimension, apply cosine to second dimension
        if samples.shape[1] > 1:
            samples[:, 1] = amplitude * torch.cos(freq * samples[:, 1])
        
        # Add some noise to make it more challenging
        samples += torch.randn_like(samples) * self.noise_std
        
        return samples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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
            (shape_ == 2.) &
            (position_x >= 0.6) &
            (position_y >= 0.6) &
            (rotation >= 2.0943) & (rotation <= 4.1888) &
            (scale < 0.6)
    )

    # dataset100 = dataset[:100]
    # mask100 = mask[:100]
    # train_data = dataset100[~mask100]
    # test_data = dataset100[mask100]
    # Apply mask to split dataset
    # train_data = [images[~mask]]  # Keep only data that do NOT match the exclusion condition
    # test_data = dataset[mask]  # Data that match the condition go into test set

    ## Save the split datasets
    np.savez("../data/2d/train.npz", imgs=images[~mask],pivot_image=pivot_image[~mask], latents_values=values[~mask])
    np.savez("../data/2d/test.npz", imgs=images[mask], pivot_image=pivot_image[mask], latents_values=values[mask])
    print("saved files")

class DspritesDataset(Dataset):
    def __init__(self, file_path):
        # Load the .npz file
        data = np.load(file_path)

        self.images = data["imgs"][:]
        # plt.figure(figsize=(5, 5))
        # plt.imshow(self.images[0].squeeze(), cmap="gray")  # Use cmap="gray" for grayscale images
        # plt.title("Varying Image")
        # plt.axis("off")
        # plt.show()
        self.latents_values = data["latents_values"][:]

        self.pivot_image = data["pivot_image"][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pivot_image = torch.tensor(self.pivot_image[idx], dtype=torch.float32)
        varying_image = torch.tensor(self.images[idx], dtype=torch.float32)
        return pivot_image, varying_image

# ============================== UTILITIES ==============================

def split_dataset_2element(train_data,test_data, batch_size=32, shuffle=True, num_workers=1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def split_dataset(dataset, train_ratio=0.7, test_ratio=0.15, batch_size=32, shuffle=True, num_workers=1):

    total_size = len(dataset)

    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    val_size = total_size - train_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, val_loader    


def get_dataloaders(dataset,
                       train_ratio, 
                       test_ratio,
                       batch_size=32, 
                       shuffle=True,
                       num_workers=1
                       ):
    
    return split_dataset(dataset,
                         train_ratio=train_ratio,
                         test_ratio=test_ratio,
                         batch_size=batch_size, 
                         shuffle=shuffle, 
                         num_workers=num_workers)

def get_dataloaders_2element(train_data,
                    test_data,
                    batch_size=32,
                    shuffle=True,
                    num_workers=1
                    ):
    return split_dataset_2element(  train_data,
                                    test_data,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers)
    

         
    
if __name__ == "__main__":
    # ds1 = SyntheticDataset(num_classes=2, n_samples_per_class=1000, x_dim=3, y_dim=64, z_dim=64, is_non_linear=True, noise_std=0.1)
    # train_loader, test_loader, val_loader = get_dataloaders(ds1, 0.7, 0.15, 32, True, 4)
    # for X, y in train_loader:
    #     print(X.shape)
    #     print(y.shape)
    #     break
    # for X, y in test_loader:
    #     print(X.shape)
    #     print(y.shape)
    #     break
    # for X, y in val_loader:
    #     print(X.shape)
    #     print(y.shape)
    #     break
    prepare_2d_data("../data/2d/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")