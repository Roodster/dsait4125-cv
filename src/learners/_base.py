

class BaseLearner:
    def __init__(self, 
                 args=None, 
                 model=None, 
                 optimizer=None,
                 criterion=None
                 ):
        self.device = args.device

        # ===== DEPENDENCIES =====
        self.args = args
        self.model = model(args).to(self.args.device)
        self.optimizer = optimizer(params=self.model.parameters(), lr=args.learning_rate)
        self.criterion = criterion()
    
    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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