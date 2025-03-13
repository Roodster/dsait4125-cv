import torch as th
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ._base import BaseLearner

    
class ClassificationLearner(BaseLearner):

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
        
        super().__init__(args=args, model=model, optimizer=optimizer, criterion=criterion)

        
    def step(self, data_loader, results):
        self.model.train()
        train_loss = .0
        
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(self.args.device), batch_labels.to(self.args.device)
        
            outputs = self.predict(batch_data)
            
            loss = self.compute_loss(y_pred=outputs, y_test=batch_labels)    
            self.update(loss=loss)

            train_loss += loss.item()

        results.train_losses = train_loss / len(data_loader)
        return results
    
    
    def evaluate(self, dataloader, results):
        self.model.eval()
        test_loss = 0.0

        metrics = {}

        with th.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                            
                outputs = self.predict(batch_data)
                    
                loss = self.compute_loss(y_pred=outputs, y_test=outputs)
                test_loss += loss.item()
                    
        class_preds = outputs.argmax(dim=1)
        metrics['accuracies'] = accuracy_score(class_preds, batch_labels)
        metrics['precisions'] = precision_score(class_preds, batch_labels)
        metrics['f1s'] = f1_score(class_preds, batch_labels)
        metrics['aucs'] = roc_auc_score(class_preds, batch_labels)

        metrics['test_losses'] = test_loss / len(dataloader)

        results.update(metrics)

        return results
