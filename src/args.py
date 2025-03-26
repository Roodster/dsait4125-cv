"""
    Args:
        Reads parameter file and returns arguments in class format.
"""

import os
import torch as th
from src.common.utils import parse_args
from datetime import datetime
class Args():
    
    def __init__(self, file):
        # ===== Get the configuration from file =====
        self.config = parse_args(file)
        
        # ===== METADATA =====
        self.exp_name = self.config.get("exp_name", "exp")
        self.mode = self.config.get("mode", "train")
        # ===== MODEL =====
        self.model_name = self.config.get("model_name", "model")
        self.latent_dim = self.config.get("latent_dim", 10)
        self.in_channels = self.config.get("in_channels", 1)
        self.img_size = self.config.get("img_size", 64)
        
        # ===== DATASET =====
        self.train_ratio = self.config.get("train_ratio", 0.7)
        self.test_ratio = self.config.get("test_ratio", 0.15)
        self.val_ratio = 1 - self.train_ratio - self.test_ratio
        
        # ===== EXPERIMENT =====
        self.seed = self.config.get("seed", 1)
        self.device = self.config.get("device", "cpu")

        # ===== TRAINING ===== 
        self.batch_size = self.config.get("batch_size", 32)
        self.n_epochs = self.config.get("n_epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 1e-3)
        
        # ===== MAGA =====
        self.beta_kl = self.config.get("beta_kl", 1)
        self.beta_recon = self.config.get("beta_recon", 1)
        
        # ===== EVALUATION =====
        self.eval_interval = self.config.get("eval_interval", 10)
        
        self.eval_save_model_interval = self.config.get("eval_save_model_interval", 10)

        self.save_all_models = self.config.get("save_all_models", False)
        
        # ===== FILE HANDLING =====
        
        self.log_dir = self.config.get("log_dir", "./outputs") + f"/run_{self.exp_name}_{self.model_name}/seed_{self.seed}_{datetime.now().strftime('%d%m%Y%H%M')}"
        self.save_model_name = self.config.get("save_model_name", "model")

        self.experiment_dir = self.config.get("experiment_dir", "")
        load_experiment_dir = len(self.experiment_dir) > 0
        # Update load_model_path to find the model file dynamically
        self.load_model_path = self.config.get("load_model_path", "")
        
        model_dir = self.experiment_dir + "/models/"

        model_files = []
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]

        self.load_model_path = os.path.join(model_dir, model_files[0]) if len(model_files) > 0 and load_experiment_dir else ""   
        
        self.checkpoint_file = self.config.get("checkpoint_file", "") 
        self.checkpoint_file = self.experiment_dir + '/stats.csv' if len(self.checkpoint_file) == 0 and load_experiment_dir else ""

        if self.load_model_path and self.checkpoint_file: 
            print(f"Continuing from checkpoint files: \n{self.checkpoint_file} \n and model: \n{self.load_model_path} ")

    def default(self):
        return self.__dict__