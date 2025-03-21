from tqdm import tqdm

from src.writer import Writer
from src.results import Results
import matplotlib.pyplot as plt

class Experiment:
    
    def __init__(self, registry, args):
            # ===== DEPENDNCIES =====
            self.args = args
            self.learner = registry['learner'](args=args, model=registry['model'], optimizer=registry['optimizer'], criterion=registry['criterion'])
            self.results = Results(verbose=False)
            self.writer = Writer(args=args)
            
            self.start_epochs = len(self.results.epochs) if self.results else None
            self.last_epoch = self.start_epochs
            
            # ===== TRAINING =====
            self.device = args.device
            self.n_epochs = args.n_epochs       
            
            
            # ===== EVALUATION =====
            assert args.eval_interval > 0, "Can't modulo by zero."
            self.eval_interval = args.eval_interval
            assert args.eval_save_model_interval > 0, "Can't modulo by zero."
            self.save_model_interval = args.eval_save_model_interval
            
            
            # ===== SEEDING =====
            self.writer.save_hyperparameters(self.args)

    def run(self, train_loader, test_loader, val_loader=None):
        assert train_loader is not None, "Please, provide a training dataset :)."
        assert test_loader is not None, "Please, provide a test dataset :)."

        # Initialize arrays to store thresholds and weights
        pbar = tqdm(range(self.last_epoch, self.last_epoch + self.n_epochs))
        
        for epoch in pbar:
            self.last_epoch += 1
            self.results = self.learner.step(train_loader, results=self.results)
            
            if (epoch + 1) % self.eval_interval == 0: 
                self.results = self.learner.evaluate(
                    data_loader=test_loader, results=self.results
                )

                self.results.epochs = epoch
                self.writer.save_model(self.learner.model, epoch=self.last_epoch)
            
        # Save the results statistics
        self.writer.save_statistics(self.results.get())
        # save the network parameters
        self.writer.save_model(self.learner.model,epoch=self.last_epoch)

    def eval(self, test_loader):
        self.results = self.learner.evaluate(
            data_loader=test_loader, results=self.results
        )
        print(self.results.test_losses)
        imgs = self.results.generated_images
        plt.figure()
        plt.imshow(imgs[0][0],cmap='gray')
        plt.show()

