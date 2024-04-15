'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: September 2023
Last updated: Adrien Dorise - September 2023

'''
import torch 
import pickle 
import cv2 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Experiment():
    def __init__(self, model,
                train_loader, test_loader, 
                num_epoch=50,
                batch_size=32,
                learning_rate=1e-03,
                weight_decay=1e-03,    
                optimizer=torch.optim.Adam,
                criterion=torch.nn.L1Loss(),
                nb_workers=0):
        #Model parameters  
        self.model         = model
        self.num_epoch     = num_epoch
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.optimizer     = optimizer
        self.criterion     = criterion
        self.nb_workers    = nb_workers
        
        self.train_loader = train_loader
        self.test_loader  = test_loader
         
    def fit(self):
        """Train the model using the data available in the train and validation folder path.
        """

        losses_train, losses_val = self.model.fit(self.train_loader,
        self.num_epoch, 
        criterion=self.criterion, 
        optimizer=self.optimizer,
        learning_rate=self.learning_rate,
        weight_decay=self.weight_decay, 
        valid_set=self.test_loader,
        loss_indicators=1)
        self.model.plotLoss(losses_train,losses_val)

    def predict(self):          
        """Model prediction on the samples available in the test folder path
        """
        # !!! Data loading !!!
        score, prediction, _ = self.model.predict(self.test_loader,self.criterion)



    def visualise(self):
        """Visualisation of the first picture of the test set.
        The input + predicted images are both shown.
        """
        # !!! Data loading !!!
        
        score, pred, (feature, target) = self.model.predict(self.test_loader,self.criterion)
        print('\nCONFUSION MATRIX : \n\n')
        cm = confusion_matrix(target, torch.argmax(torch.tensor(pred), dim=1))
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig('/home/dr-evil/dev_dragonflAI/history/dragonflai/examples/VIT_MNIST/results/CM.png')
        
        
    
    def save(self, filename):
        """Save the whole experiment class as a pickle object.

        Args:
            filename (string): Path to save the experiment status
        """
        with open(filename, 'wb') as file:
            try:
                pickle.dump(self, file)
            except EOFError:
                raise Exception("Error in save experiment: Pickle was not able to save the file.")

    @classmethod
    def load(self, filename):
        """Load a pickle object to an Experiment class Python variable
        This is a class method. It means that a reference to the class is NOT necessary to call this method. Simply type <your_experiment = Experiment.load(filename)>

        Args:
            filename (string): Path to the pickle saved object.
        """
        with open(filename, 'rb') as file:
            try:
               return pickle.load(file)
            except EOFError:
                raise Exception("Error in load experiment: Pickle was not able to retrieve the file.")
