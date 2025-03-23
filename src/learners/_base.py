import torch

class BaseLearner:
    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None
                 ):

        assert args is not None, "No args defined."
        assert model is not None, "No model defined."
        assert optimizer is not None, "No optimizer defined."
        assert criterion is not None, "No criterion defined."
        
        self.device = args.device

        # ===== DEPENDENCIES =====
        self.args = args
        self.model = model(args).to(self.args.device)
        
        if len(args.load_model_path) > 0:
            self.model.load_state_dict(torch.load(args.load_model_path, weights_only=True))

        self.optimizer = optimizer(params=self.model.parameters(), lr=args.learning_rate)

        self.n_updates = 0
        self.n_epochs = args.n_epochs
        
        if self.args.model_name in  ("maga","vae"):
            self.criterion = criterion(args)
        else:
            self.criterion = criterion()
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.n_updates += 1
        
    def predict(self, batch_data):
        outputs = self.model(batch_data)
        return outputs

    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def compute_loss(self, y_pred, y_test):
        loss =  self.criterion(y_pred, y_test)
        return loss