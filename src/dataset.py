import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class BinarySyntheticDataset(Dataset):
    def __init__(self, num_classes=2, n_samples_per_class=1000, img_size=64, 
                 pattern_type='geometric', seed=42, noise_prob=0.05):
        """
        Create a synthetic dataset of binary images with classifiable patterns.
        
        Args:
            num_classes: Number of different classes/patterns to generate
            n_samples_per_class: Number of samples per class
            img_size: Size of the square images (img_size x img_size)
            pattern_type: Type of pattern ('geometric', 'frequency', 'random')
            seed: Random seed for reproducibility
            noise_prob: Probability of flipping a pixel (noise)
        """
        self.num_classes = num_classes
        self.n_samples_per_class = n_samples_per_class
        self.img_size = img_size
        self.pattern_type = pattern_type
        self.seed = seed
        self.noise_prob = noise_prob
        
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Generate the data
        self.data, self.labels = self._generate_data()
        
    def _generate_data(self):
        data = []
        labels = []
        
        for class_idx in range(self.num_classes):
            # Generate samples for this class
            class_samples = self._generate_class_samples(class_idx)
            
            # Add noise (randomly flip pixels with probability noise_prob)
            noise_mask = torch.rand_like(class_samples) < self.noise_prob
            class_samples[noise_mask] = 1 - class_samples[noise_mask]
            
            data.append(class_samples)
            
            # Create labels for this class
            class_labels = torch.full((self.n_samples_per_class,), class_idx, dtype=torch.long)
            labels.append(class_labels)
        
        # Concatenate all classes
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return data, labels
    
    def _generate_class_samples(self, class_idx):
        """Generate binary patterns for a specific class"""
        samples = torch.zeros(self.n_samples_per_class, 1, self.img_size, self.img_size)
        
        if self.pattern_type == 'geometric':
            # Generate geometric patterns (circles, squares, etc.)
            for i in range(self.n_samples_per_class):
                # Vary the pattern slightly for each sample
                variation = (i / self.n_samples_per_class) * 0.5
                samples[i] = self._create_geometric_pattern(class_idx, variation)
                
        elif self.pattern_type == 'frequency':
            # Generate frequency-based patterns
            for i in range(self.n_samples_per_class):
                # Vary the frequency slightly for each sample
                variation = (i / self.n_samples_per_class) * 0.5
                samples[i] = self._create_frequency_pattern(class_idx, variation)
                
        elif self.pattern_type == 'random':
            # Generate random but consistent patterns for each class
            for i in range(self.n_samples_per_class):
                samples[i] = self._create_random_pattern(class_idx, i)
        
        # Ensure binary values (0 or 1) as float tensors
        samples = (samples > 0.5).float()
        
        return samples
    
    def _create_geometric_pattern(self, class_idx, variation):
        """Create geometric patterns like circles, squares based on class"""
        img = torch.zeros(1, self.img_size, self.img_size)
        center = self.img_size // 2
        
        # Different pattern for each class
        if class_idx % 4 == 0:  # Circle
            radius = int(center * (0.5 + variation))
            for i in range(self.img_size):
                for j in range(self.img_size):
                    if (i - center)**2 + (j - center)**2 < radius**2:
                        img[0, i, j] = 1
                        
        elif class_idx % 4 == 1:  # Square
            size = int(center * (0.5 + variation))
            start = center - size
            end = center + size
            img[0, start:end, start:end] = 1
            
        elif class_idx % 4 == 2:  # Cross
            thickness = int(self.img_size * (0.1 + variation * 0.1))
            half_thick = thickness // 2
            img[0, center-half_thick:center+half_thick+1, :] = 1  # Horizontal
            img[0, :, center-half_thick:center+half_thick+1] = 1  # Vertical
            
        else:  # Diamond
            size = int(center * (0.5 + variation))
            for i in range(self.img_size):
                for j in range(self.img_size):
                    if abs(i - center) + abs(j - center) < size:
                        img[0, i, j] = 1
        
        return img
    
    def _create_frequency_pattern(self, class_idx, variation):
        """Create patterns based on different frequencies"""
        img = torch.zeros(1, self.img_size, self.img_size)
        
        # Base frequency depends on class
        freq_x = 1 + class_idx % 3 + variation
        freq_y = 1 + (class_idx // 3) % 3 + variation
        
        # Create grid of coordinates
        x = torch.linspace(-np.pi, np.pi, self.img_size)
        y = torch.linspace(-np.pi, np.pi, self.img_size)
        xv, yv = torch.meshgrid(x, y, indexing='ij')
        
        # Generate pattern based on sine waves
        if class_idx % 2 == 0:
            # Sine wave pattern
            pattern = torch.sin(freq_x * xv) * torch.sin(freq_y * yv)
        else:
            # Checkerboard-like pattern
            pattern = torch.sin(freq_x * xv) * torch.cos(freq_y * yv)
        
        # Convert to binary
        img[0] = (pattern > 0).float()
        
        return img
    
    def _create_random_pattern(self, class_idx, sample_idx):
        """Create random but consistent patterns for each class"""
        # Set a seed based on class and sample for consistency
        local_seed = self.seed + class_idx * 10000 + sample_idx
        torch.manual_seed(local_seed)
        
        # Create a base random pattern (smaller resolution)
        base_pattern = torch.rand(1, self.img_size // 4, self.img_size // 4)
        base_pattern = (base_pattern > 0.5).float()
        
        # Upsample to full size
        img = torch.nn.functional.interpolate(
            base_pattern.unsqueeze(0), 
            size=(self.img_size, self.img_size),
            mode='nearest'
        ).squeeze(0)
        
        return img
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx % len(self.data)] 

class DspritesDataset(Dataset):
    def __init__(self, file_path, single_output):
        # Load the .npz file
        data = np.load(file_path)
        self.single_output = single_output
        self.images = data["imgs"][:]
        # plt.figure(figsize=(5, 5))
        # plt.imshow(self.images[0].squeeze(), cmap="gray")  # Use cmap="gray" for grayscale images
        # plt.title("Varying Image")
        # plt.axis("off")
        # plt.show()
        self.latents_values = data["latents_values"][:]

        self.pivot_image = data["pivot_image"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        p_img = self.pivot_image
        v_img = self.images[idx]
        pivot_image = torch.tensor(p_img.reshape(1, 64, 64), dtype=torch.float32)
        varying_image = torch.tensor(v_img.reshape(1, 64, 64), dtype=torch.float32)

        return varying_image if self.single_output else (pivot_image, varying_image)

# ============================== UTILITIES ==============================

def split_dataset_2element(train_data,test_data, batch_size=32, shuffle=True, num_workers=1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
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
    pass
