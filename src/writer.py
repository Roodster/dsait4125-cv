"""
    Writer:
        Writes data to files.
"""
import os
import torch as th
import matplotlib.pyplot as plt
import json
import datetime as dt
import yaml
import pandas as pd

class Writer():
    """
        Based on folder structure:
            ./outputs
               ./exp_<exp-name>_<model>
                    ./seed_<seed>_<datetime>
                        ./models
                    ./evaluation_<exp-name>_<model>
                    
    """
        
    def __init__(self, args):
        
        self.args = args
        # self.datetime = str(dt.datetime.now().strftime("%d%m%Y%H%M"))
        self.root = args.log_dir
        self.eval_dir = self.root + f"/evaluation"
        self.model_dir = self.root + "/models"
        self._create_directories(self.root)
        self._create_directories(self.eval_dir)
        self._create_directories(self.model_dir)

        self.save_all_models = args.save_all_models
        self.save_epoch = 0

    def _create_directories(self, path):
        do_exist = os.path.exists(path)
        if not do_exist:
            # Create a new directory because it does not exist
            os.makedirs(path)
    
    def save_model(self, model, epoch):
        _dir = os.path.join(self.model_dir)
        # _name = self.args.save_model_name
        postfix = f"_epoch_{self.save_epoch}" if self.save_all_models else f""
        file = f"/{self.args.save_model_name}{postfix}.pth"

        full_path = _dir + file
        th.save(model.state_dict(), full_path)
    
    def save_plot(self, plot, attribute):
        filepath = f"/plot_{attribute}.png"
        
        plot_path = self.eval_dir + filepath
        
        plot.savefig(plot_path)
    
    def save_statistics(self, statistics):
        filepath = f"/stats.csv"
        stats_path = self.root + filepath
        statistics.to_csv(stats_path, index=False)
            
    def save_hyperparameters(self, hyperparameters):
        filepath = f"/hyperparameters.yaml"

        hyperparams_path = self.root + filepath
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparameters.__dict__,
                      f,
                      indent=4,
                      sort_keys=True,
                      default=str)
