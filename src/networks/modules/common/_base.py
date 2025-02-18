import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import inspect

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
        self._modules_initialized = False
        self.to(self.device)
        
    def __setattr__(self, name, value):
        if isinstance(value, (torch.Tensor, nn.Module)) and hasattr(self, 'device'):
            value = value.to(self.device)
        super().__setattr__(name, value)
    
    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get('device', self.device)
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        self.device = device
        
        # Move all tensors and submodules to device
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, (torch.Tensor, nn.Module)):
                setattr(self, attr_name, attr_value.to(device))
        
        self._modules_initialized = True
        return super().to(device)

    @staticmethod
    def _convert_to_tensor(arr):
        if arr is None:
            return None
        
        if isinstance(arr, torch.Tensor):
            return arr
            
        dtype_map = {
            np.float64: torch.float64,
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int64: torch.long,
            np.int32: torch.int,
            np.int16: torch.short,
            np.uint8: torch.uint8,
            np.bool_: torch.bool
        }
        
        if isinstance(arr, np.ndarray):
            torch_dtype = dtype_map.get(arr.dtype.type, torch.float32)
            tensor = torch.tensor(arr, dtype=torch_dtype)
            return tensor
        return arr
    
    def _numpy_input_wrapper(self, func):
        @functools.wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Convert args and move to device
            converted_args = []
            for arg in args:
                tensor = self._convert_to_tensor(arg)
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(instance.device)
                converted_args.append(tensor)
            
            # Convert kwargs and move to device
            converted_kwargs = {}
            for k, v in kwargs.items():
                tensor = self._convert_to_tensor(v)
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.to(instance.device)
                converted_kwargs[k] = tensor
            
            result = func(instance, *converted_args, **converted_kwargs)
            return result
        return wrapper

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Get all methods that need wrapping
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_') or name == '__init__':
                wrapped = cls._numpy_input_wrapper(cls, method)
                setattr(cls, name, wrapped)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")