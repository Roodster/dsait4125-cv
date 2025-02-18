from torch.utils.data import DataLoader

import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


from src.args import Args
from src.registry import setup
from src.dataset import get_dataloaders, SyntheticDataset
from src.experiment import Experiment
from src.common.utils import set_seed

def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)

    # load dataset
    dataset = SyntheticDataset(num_classes=2, n_samples_per_class=128, x_dim=3, y_dim=64, z_dim=64)
    train_loader, test_loader, val_loader = get_dataloaders(dataset=dataset, 
                                                            train_ratio=args.train_ratio, 
                                                            test_ratio=args.test_ratio,
                                                            batch_size=args.batch_size)

    # initialize experiment
    experiment = Experiment(registry=registry, 
                            args=args 
                            )
    
    # run experiment
    experiment.run(train_loader=train_loader, test_loader=test_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()