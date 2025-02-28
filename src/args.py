"""
    Args:
        Reads parameter file and returns arguments in class format.
"""

import torch as th
from src.common.utils import parse_args
from datetime import datetime
class Args():
    
    def __init__(self, file):
        # ===== Get the configuration from file =====
        self.config = parse_args(file)
        
        # ===== METADATA =====
        self.exp_name = self.config.get("exp_name", "exp")
        
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
        
        # ===== FILE HANDLING =====
        self.log_dir = self.config.get("log_dir", "./outputs") + f"/run_{self.exp_name}_{self.model_name}/seed_{self.seed}_{datetime.now().strftime('%d%m%Y%H%M')}"
        
    def default(self):
        return self.__dict__