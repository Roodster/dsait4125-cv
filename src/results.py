import pandas as pd
import numpy as np
from pprint import pprint

class Results:
    """
    Class to store and manage experiment results.
    Handles metrics with different lengths and ensures only raw values are stored.
    """
    def __init__(self, file=None, verbose=False):
        self._prev_results = None
        self._results = None
        self.verbose = verbose
        
        # Initialize metric lists
        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._accuracies = []
        self._sensitivities = []
        self._precisions = []
        self._aucs = []
        self._f1s = []
        self._train_kl_losses = []
        self._train_recon_losses = []
        self._train_recon_latent_losses = []
        self._test_recon_losses = []
        self._test_recon_latent_losses = []
        self._test_kl_losses = []

        self._generated_images = []
        
        # If a file is provided, read the data and populate the lists
        if file is not None:
            self._prev_results = pd.read_csv(file)
            
            # Populate the lists with data from the dataframe
            for column in self._prev_results.columns:
                attr_name = f"_{column}"
                if hasattr(self, attr_name):
                    setattr(self, attr_name, self._prev_results[column].tolist())

    def get(self):
        """Returns a DataFrame with all metrics that have the same length."""
        results_dict = self._get_valid_metrics()
        self._results = pd.DataFrame(results_dict)
        return self._results
    
    def _get_valid_metrics(self):
        """Returns a dictionary with metrics that have the same length."""
        # Get all attribute names that start with underscore and are lists
        metric_attrs = {attr: getattr(self, attr) for attr in dir(self) 
                       if attr.startswith('_') and 
                       isinstance(getattr(self, attr), list) and
                       attr != "_generated_images"}
        
        # Find the most common length among non-empty lists
        lengths = [len(val) for val in metric_attrs.values() if len(val) > 0]
        if not lengths:
            return {}
            
        # Get the most common length
        target_len = max(set(lengths), key=lengths.count)
        
        # Filter attributes to only include those with the target length
        valid_metrics = {}
        for attr, values in metric_attrs.items():
            if len(values) == target_len:
                # Remove the leading underscore for the key
                key = attr[1:] if attr.endswith('s') else attr[1:] + 's'
                valid_metrics[key] = values
                
        return valid_metrics
    
    def print(self):
        """Print the valid metrics."""
        pprint(self._get_valid_metrics())

    def update(self, updates):
        """Updates attributes based on the provided dictionary."""
        for attr_name, value in updates.items():
            # Convert to the internal attribute name format
            internal_name = f"_{attr_name}"
            
            # If attribute doesn't exist, create it
            if not hasattr(self, internal_name):
                setattr(self, internal_name, [])
                if self.verbose:
                    print(f"Created new metric: {attr_name}")
            
            # Get the current value
            current_value = getattr(self, internal_name)
            
            # Ensure we're storing raw values, not tensors
            if hasattr(value, 'item'):
                value = value.item()
            
            # Update the attribute
            if isinstance(current_value, list):
                current_value.append(value)
            else:
                setattr(self, internal_name, value)


    # Property and setter for epochs
    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._epochs.append(value)

    # Property and setter for train_losses
    @property
    def train_losses(self):
        return self._train_losses 
        
    
    @train_losses.setter
    def train_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._train_losses.append(value)
        
    # Property and setter for recon_losses
    @property
    def train_recon_losses(self):
        return self._test_recon_losses
    
    @train_recon_losses.setter
    def train_recon_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._train_recon_losses.append(value)
        
    # Property and setter for recon_latent_losses
    @property
    def train_recon_latent_losses(self):
        return self._train_recon_latent_losses
    
    @train_recon_latent_losses.setter
    def train_recon_latent_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._train_recon_latent_losses.append(value)
        
    # Property and setter for kl_losses
    @property
    def train_kl_losses(self):
        return self._train_kl_losses
    
    @train_kl_losses.setter
    def train_kl_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._train_kl_losses.append(value)
        
    # Property and setter for test_losses
    @property
    def test_losses(self):        
        return self._test_losses 
    
    @test_losses.setter
    def test_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._test_losses.append(value)

    # Property and setter for recon_losses
    @property
    def test_recon_losses(self):
        return self._test_recon_losses
    
    @test_recon_losses.setter
    def test_recon_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._test_recon_losses.append(value)
        
    # Property and setter for recon_latent_losses
    @property
    def test_recon_latent_losses(self):
        return self._test_recon_latent_losses
    
    @test_recon_latent_losses.setter
    def test_recon_latent_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._test_recon_latent_losses.append(value)
        
    # Property and setter for kl_losses
    @property
    def test_kl_losses(self):
        return self._test_kl_losses
    
    @test_kl_losses.setter
    def test_kl_losses(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._test_kl_losses.append(value)

    @property
    def generated_images(self):
        return self._generated_images

    @generated_images.setter
    def generated_images(self, value):
        # if hasattr(value, 'item'):
        #     value = value
        self._generated_images.append(value)

    # Property and setter for accuracies
    @property
    def accuracies(self):
        return self._accuracies
    
    @accuracies.setter
    def accuracies(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._accuracies.append(value)

    # Property and setter for sensitivities
    @property
    def sensitivities(self):
        return self._sensitivities
    
    @sensitivities.setter
    def sensitivities(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._sensitivities.append(value)

    # Property and setter for precisions
    @property
    def precisions(self):
        return self._precisions
    
    @precisions.setter
    def precisions(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._precisions.append(value)

    # Property and setter for aucs
    @property
    def aucs(self):
        return self._aucs
    
    @aucs.setter
    def aucs(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._aucs.append(value)

    # Property and setter for f1s
    @property
    def f1s(self):
        return self._f1s
    
    @f1s.setter
    def f1s(self, value):
        # Ensure we're storing raw values, not tensors
        if hasattr(value, 'item'):
            value = value.item()
        self._f1s.append(value)