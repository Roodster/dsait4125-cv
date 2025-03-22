import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from src.args import Args
from src.registry import setup
from src.dataset import DspritesDataset, get_dataloaders_2element, BinarySyntheticDataset, get_dataloaders
from src.experiment import Experiment
from src.common.utils import set_seed

def main():
        
    # Load arguments
    args = Args(file="./data/configs/default.yaml")
    registry = setup(args.model_name)
    set_seed(args.seed)
    
    # Determine whether to use single_output based on model type
    single_output = args.model_name == "vae"

    # Load dataset with the correct output format
    train_data = DspritesDataset("./data/2d/train.npz", single_output=single_output)
    test_data = DspritesDataset("./data/2d/test.npz", single_output=single_output)

    # Choose the correct data loader function
    if args.model_name == "maga":
        train_loader, test_loader = get_dataloaders_2element(
            train_data, test_data,
            batch_size=args.batch_size
        )
    elif args.model_name == "vae":
        train_loader, test_loader, val_loader = get_dataloaders(
            dataset=train_data + test_data  ,  # Only passing train_data here
            train_ratio=args.train_ratio,
            test_ratio=args.test_ratio,
            batch_size=args.batch_size
        )

    # dataset = BinarySyntheticDataset(num_classes=3, n_samples_per_class=100, img_size=64, pattern_type='geometric', seed=42, noise_prob=0.05)
    # train_loader, test_loader, val_loader = get_dataloaders(dataset=dataset, train_ratio=0.7, test_ratio=0.2, batch_size=32)

    # initialize experiment
    experiment = Experiment(registry=registry, 
                            args=args 
                            )
    
    # run experiment
    # experiment.run(train_loader=train_loader, test_loader=test_loader)
    experiment.eval(test_loader)



if __name__ == "__main__":
    main()