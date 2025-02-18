import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


# ============================== UTILITIES ==============================


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
    
    

         
    
if __name__ == "__main__":
    ds1 = SyntheticDataset(num_classes=2, n_samples_per_class=1000, x_dim=3, y_dim=64, z_dim=64, is_non_linear=True, noise_std=0.1)
    train_loader, test_loader, val_loader = get_dataloaders(ds1, 0.7, 0.15, 32, True, 4)
    for X, y in train_loader:
        print(X.shape)
        print(y.shape)
        break
    for X, y in test_loader:
        print(X.shape)
        print(y.shape)
        break
    for X, y in val_loader:
        print(X.shape)
        print(y.shape)
        break