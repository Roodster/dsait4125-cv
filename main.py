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
        # train_loader, test_loader, val_loader = get_dataloaders(
        #     dataset=train_data + test_data  ,  # Only passing train_data here
        #     train_ratio=args.train_ratio,
        #     test_ratio=args.test_ratio,
        #     batch_size=args.batch_size
        # )
        train_loader, test_loader = get_dataloaders_2element(
            train_data, test_data,
            batch_size=args.batch_size
        )

    # initialize experiment
    experiment = Experiment(registry=registry, 
                            args=args 
                            )
    
    # run experiment
    if args.mode == 'train':
        experiment.run(train_loader=train_loader, test_loader=test_loader)
    elif args.mode == 'eval':
        experiment.eval(test_loader)



if __name__ == "__main__":
    main()