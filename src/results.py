import pandas as pd
import numpy as np

from pprint import pprint

class Results:
    def __init__(self, file=None, verbose=False):
        self._prev_results = None
        self._results = None
        self.verbose = verbose
        # Initialize lists
        self._epochs = []
        self._train_losses = []
        self._test_losses = []
        self._accuracies = []
        self._sensitivities = []
        self._precisions = []
        self._aucs = []
        self._f1s = []
        # If a file is provided, read the data and populate the lists
        if file is not None:
            self._prev_results = pd.read_csv(file)
            
            # Populate the lists with data from the dataframe
            self._epochs = self._prev_results['epoch'].tolist()
            self._train_losses = self._prev_results['train_loss'].tolist()
            self._test_losses = self._prev_results['test_loss'].tolist()
            self._accuracies = self._prev_results['accuracy'].tolist()
            self._precisions = self._prev_results['precision'].tolist()
            self._aucs = self._prev_results['auc'].tolist()
            self._f1s = self._prev_results['f1'].tolist()

    def get(self):

        results = self._get()
        self._results = pd.DataFrame(results)

        return self._results
    
    def _get(self):
        return {
                'epoch': self._epochs,
                'train_loss': self._train_losses,
                'test_loss': self._test_losses,
                'accuracy': self._accuracies,
                'precision': self._precisions,
                'auc': self._aucs,
                'f1': self._f1s,
            }
    
    def print(self):
        pprint(self._get())

    def update(self, updates):
        """Updates attributes based on the provided dictionary."""
        for attr_name, value in updates.items():
            if hasattr(self, "_" + attr_name):  # Check if the attribute exists
                current_value = getattr(self, "_" + attr_name)
                if isinstance(current_value, list): # If it's a list, append
                    current_value.append(value)
                else:                               # Otherwise, directly set
                    setattr(self, "_" + attr_name, value)
            else:
                print(f"Warning: Attribute '{attr_name}' not found in Results object.")


    # Property and setter for train_losses
    @property
    def train_losses(self):
        return self._train_losses 
    
    @train_losses.setter
    def train_losses(self, value):
        self._train_losses.append(value)

    # Property and setter for test_losses
    @property
    def test_losses(self):        
        return self._test_losses 
    
    @test_losses.setter
    def test_losses(self, value):
        self._test_losses.append(value)

    # Property and setter for epochs
    @property
    def epochs(self):
        return self._epochs
    
    @epochs.setter
    def epochs(self, value):
        self._epochs.append(value)

    # Property and setter for accuracies
    @property
    def accuracies(self):
        if self.verbose:
            print('accuracies: \n', self._accuracies)
        return self._accuracies
    
    @accuracies.setter
    def accuracies(self, value):
        self._accuracies.append(value)

    # Property and setter for sensitivities
    @property
    def sensitivities(self):
        return self._sensitivities
    
    @sensitivities.setter
    def sensitivities(self, value):
        self._sensitivities.append(value)

    # Property and setter for precisions
    @property
    def precisions(self):
        return self._precisions
    
    @precisions.setter
    def precisions(self, value):
        self._precisions.append(value)

    # Property and setter for aucs
    @property
    def aucs(self):
        return self._aucs
    
    @aucs.setter
    def aucs(self, value):
        self._aucs.append(value)

    # Property and setter for accuracy (overall accuracy)
    @property
    def f1s(self):
        return self._f1s
    
    @f1s.setter
    def f1s(self, value):
        self._f1s.append(value)
        