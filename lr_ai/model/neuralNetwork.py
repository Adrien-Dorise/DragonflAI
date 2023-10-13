"""
This package references all neural network classes used in the application.
Author: Julia Cohen - Adrien Dorise - Edouard Villain ({jcohen, adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Adrien Dorise - August 2023

Pytorch is the main API used.
It is organised as follow:
    - NeuralNetwork class: Core class that contains all tools to use a neural network (training, testing, print...)
    - Subsidiary net classes: This class has to be setup by the user. They contains the information about the architecture used for each custom networks.
    
The package works as follow:
    - Use or create a neural network class.
    - Use Sequential.add_modules() to add each layer of the network
    - Available layer type: Conv2d, MaxPool2d, Linear, CrossEntropyLoss, MSELoss, ReLU, Sigmoid, Softmax, Flatten...
    - Available classes: 1) ConcolutionalNN = Convolutional + fully connected network -> image input = (n° channels, width, heidth)
                         2) fullyConnectedNN = Fully connected network -> input = (int)
"""

from os.path import exists
from tqdm import tqdm

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.cuda import is_available
from torchinfo import summary
from torchview import draw_graph

import torch.nn as nn
import torch
from torch.cuda import amp  
import numpy as np
import matplotlib.pyplot as plt
import time 
import os 

 


class NeuralNetwork(nn.Module):
    """Main Pytorch neural network class. 
    
    It acts as a superclass for which neural network subclasses that will describe specific architectures.
    Contains all tools to train, predict, save... a neural network model.
    
    Parameters:
        device (torch.device): sets the workload on CPU or GPU if available
        architecture (torch.nn.Sequential): Contains neural network model
           
    Use example:
        model = nnSubclass(input_size)
        model.printArchitecture((1,input_size))
        model.fit(trainset, epoch)
        score = model.predict(testset)
        print(f"Test loss: {score}")
    """
    
    def __init__(self, inputs=1023, outputs=2):
        super().__init__()
        self.outputs    = outputs
        self.inputs     = inputs
        self.duration_t = 0
        self.istrain    = False
        self.verbosity  = 1
        
        #If available -> work on GPU
        self.device = torch.device('cuda:0' if is_available() else 'cpu')
        print(f"Pytorch is setup on: {self.device}")
        self.architecture = nn.Sequential().to(self.device)

        self.model_name =  "NeuralNetwork"
        
    def _on_epoch_start(self, *args, **kwargs):
        '''callback function, called at each epoch's start'''
        self.architecture.train() 
        self.current_batch_train = 0
        
    def _on_predict_start(self, *args, **kwargs):
        '''callback function, called at each predict start'''
        self.architecture.eval()
        self.current_batch_test = 0
        
    def _on_predict_end(self, *args, **kwargs):
        '''callback function, called at each predict end'''
        pass 
    
    def _on_epoch_end(self, *args, **kwargs):
        '''callback function, called at each epoch's end'''
        pass 
    
    def _on_batch_start(self, *args, **kwargs):
        '''callback function, called at each batch's start'''
        self.start_t = time.time()
    
    def _on_batch_end(self, *args, **kwargs):
        '''callback function, called at each batch's end'''
        self.duration_t = time.time() - self.start_t
    
    def _on_training_start(self, *args, **kwargs):
        '''callback function, called at training start'''
        print('\tStart training...') 
    
    def _on_training_end(self, *args, **kwargs):
        '''callback function, called at training end'''
        print('\tEnd training...')

    def init_results(self, loss_indicators, train_loader, test_loader, batch_size, epochs, *args, **kwargs):
        self.use_gpu=False
        #Use GPU if available
        if torch.cuda.is_available():
            print("CUDA compatible GPU found")
            self.use_gpu=True
        else:
            print("No CUDA compatible GPU found")
        
        self.all_loss              = []
        self.losses_train          = [[] for _ in range(loss_indicators)]
        self.losses_val            = []
        self.dataset_size          = len(train_loader) * batch_size
        self.steps_per_epoch_train = len(train_loader)
        self.steps_per_epoch_test  = len(test_loader)
        self.epochs                = epochs
        self.batch_size            = batch_size

    def set_optimizer(self, 
                      optimizer=AdamW, 
                      learning_rate=1e-4, 
                      weight_decay=1e-4, 
                      *args, **kwargs):
        '''set optimizer'''
        self.opt = [None]
        if weight_decay is None:
            self.opt[0] = optimizer(self.architecture.parameters(), lr=learning_rate)
        else:
            self.opt[0] = optimizer(self.architecture.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
        
        self.scaler = [None]
        self.scaler[0] = amp.GradScaler(enabled=self.use_gpu)
        
    def set_scheduler(self, scheduler=None, *args, **kwargs):
        '''set scheduler'''
        self.scheduler = [None]
        
        if self.scheduler is not None:
            self.scheduler[0] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt[0], mode='min', factor=0.75, patience=10) 
        
    def update_scheduler(self, *args, **kwargs):
        '''update scheduler'''
        loss = kwargs['loss']
        if(self.scheduler[0] is not None):
            for scheduler in self.scheduler:
                scheduler.step(loss)

    def init_epoch(self, *args, **kwargs):
        '''init epoch'''
        loss_indicators   = kwargs['loss_indicators']
        self.running_loss = [0 for _ in range(loss_indicators)]
        for opt in self.opt:
            opt.zero_grad()
        self.batch_loss    = [[] for _ in range(loss_indicators)]
        self.current_batch_train = 0
        self.current_batch_test = 0
    
    def get_batch(self, *args, **kwargs):
        '''get batch'''
        sample = kwargs['sample']
        return sample[0].to(self.device), sample[1].to(self.device)
            
    def loss_calculation(self, crit, inputs, target, *args, **kwargs):
        '''compute loss'''
        
        outputs = self.forward(inputs)
        loss    = crit(outputs, target)
        
        return loss, outputs

    def update_train_loss(self, loss, *args, **kwargs):
        '''Update the loss during train'''
        inputs = kwargs["inputs"]
        
        for i in range(len(self.batch_loss)):
            self.batch_loss[i].append([loss][i].cpu().item())
            self.running_loss[i] += [loss][i].item() * inputs.size(0)                

    def train_batch(self, *args, **kwargs):
        '''train a batch '''
        loss = kwargs['loss']
        
        #See here for detail about multiple scaler & optimizer
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
        for idx, scaler in enumerate(self.scaler):
            retain_graph = (idx < len(self.scaler)-1)    
            scaler.scale(loss).backward(retain_graph=retain_graph)    
        
        for scaler in self.scaler:
            for opt in self.opt:
                scaler.step(opt)

        for scaler in self.scaler:
            scaler.update()
        
        for opt in self.opt:
            opt.zero_grad()

        self.current_batch_train += 1

    def plot_log(self, *args, **kwargs):
        '''plot log during training'''
        
        if self.verbosity > 0:
            epoch    = kwargs['epoch']
            loss     = kwargs['loss']
            val_loss = kwargs['val_loss']
            lr       = kwargs['lr']
            
            column, _ = os.get_terminal_size()
            verbose = self.verbosity
            if self.current_batch_test == self.steps_per_epoch_test:
                verbose = 2
                
            if not self.istrain:
                lr = 0
                    
            size_bar = column - 115
            i        = (size_bar * (self.current_batch_train + self.current_batch_test) // (self.steps_per_epoch_train + self.steps_per_epoch_test))
            end      = '\r'
            total    = (self.steps_per_epoch_train + self.steps_per_epoch_test) 
            if self.current_batch_test == self.steps_per_epoch_test:
                est_t = 't ~ {} s.'.format(np.around(self.duration_t * total, decimals=2))
            else:
                est_t  = 't ~ {} s.'.format(np.around(self.duration_t * (total - (self.current_batch_train + self.current_batch_test)), decimals=2))
            
            if verbose == 2:
                end = '\n'
            print('[{:4d}/{:4d}, {:4d}/{:4d}, {:4d}/{:4d}] : [{}>{}] : lr = {:.3e} - loss = {:.3e} - val = {:.3e} - {}  '.format(epoch,self.epochs,
                                                                                            self.current_batch_train, self.steps_per_epoch_train,
                                                                                            self.current_batch_test, self.steps_per_epoch_test,
                                                                                            '=' * i, ' ' * (size_bar - i),
                                                                                            lr, loss, val_loss, est_t), end=end)

    def save_epoch_end(self, *args, **kwargs):
        epoch           = kwargs['epoch']
        loss_indicators = kwargs['loss_indicators']
        dataset_size    = kwargs['dataset_size']
        
        #if epoch == 0: #Print network architecture
        #    draw_graph(self.architecture, input_data=inputs, save_graph=True,directory="models/tmp/")

        self.epoch_losses = [[] for _ in range(loss_indicators)]
        for i in range(loss_indicators):
            self.epoch_losses[i] = self.running_loss[i] / dataset_size
            self.losses_train[i].append(self.epoch_losses[i])        
            
        if epoch % 100 == 0: #Save model every X epochs
            self.saveModel(f"models/tmp/epoch{epoch}")
            
        try:  
            if self.losses_train[0][-1] == np.min(self.losses_train[0]):
                self.saveModel("./models/tmp/{}_best_train".format(self.model_name, epoch))
            if self.losses_val[-1] == np.min(self.losses_val):
                self.saveModel("./models/tmp/{}_best_val".format(self.model_name, epoch))
        except:
            pass 
                
        
    def update_outputs(self):
        """Change the last layer of the network to match the desired number of outputs.
        
        """
        assert isinstance(self.architecture, nn.Sequential), \
            "update_outputs needs to be overriden if architecture is not a Sequential module"
        in_features = self.architecture[-1].in_features
        bias = self.architecture[-1].bias is not None
        new_fc = nn.Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture[-1] = new_fc.to(self.device)
    
    def fit(self, train_set, epochs, 
            criterion=nn.L1Loss(), 
            optimizer=Adam, 
            learning_rate=0.001, 
            weight_decay=None, 
            valid_set=None,
            loss_indicators=1, 
            batch_size=2):
        """Train a model on a training set
        print(f"Pytorch is setup on: {self.device}")

        
        Args:
            train_set (torch.utils.data.DataLoader): Training set used to fit the model. This variable contains batch size information + features + target 
            epochs (int): Amount of epochs to perform during training
            criterion (torch.nn): Criterion used during training for loss calculation (default = L1Loss() - see: https://pytorch.org/docs/stable/nn.html#loss-functions) 
            optimizer (torch.optim): Optimizer used during training for backpropagation (default = Adam - see: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
            learning_rate: learning_rate used during backpropagation (default = 0.001)
            valid_set (torch.utils.data.DataLoader): Validation set used to verify model learning. Not mandatory (default = None)
            loss_indicators (int): Number of loss indicators used during training. Most NN only need one indicator, but distillation models need three (loss, loss_trainer, loss_student). (default = 1)
        """
        self.istrain = True
        self._on_training_start()
        self.init_results(loss_indicators, train_set, valid_set, batch_size, epochs)
        self.set_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        self.set_scheduler(None)
        
        for epoch in range(epochs):
            self._on_epoch_start()
            self.init_epoch(loss_indicators=loss_indicators)
            self.plot_log(epoch=epoch, loss=0, val_loss=0, lr=self.opt[0].param_groups[0]['lr'])
            for batch_ndx, sample in enumerate(train_set):
            #for i in range(self.steps_per_epoch):
                self._on_batch_start()
                inputs, targets = self.get_batch(sample=sample)
                loss,_ = self.loss_calculation(criterion, inputs, targets, loss_indicators=loss_indicators)
                self.update_train_loss(loss, inputs=inputs)
                
                self.train_batch(loss=loss)
                self.update_scheduler(loss=loss)
                self.plot_log(epoch=epoch, loss=np.mean(self.batch_loss), val_loss=0, lr=self.opt[0].param_groups[0]['lr'])
                self._on_batch_end()
                
            self.save_epoch_end(epoch=epoch, loss_indicators=loss_indicators, dataset_size=self.dataset_size)
            self._on_epoch_end()
            val_loss, _, _ = self.predict(valid_set, crit=criterion, loss_indicators=1, epoch=epoch, train_loss=np.mean(self.batch_loss))
            self.losses_val.append(val_loss)
        
        self._on_training_end()
        self.istrain = False
        return self.losses_train, self.losses_val


    def predict(self, test_set, crit=nn.L1Loss(), loss_indicators=1, epoch=0, train_loss=0):
        """Use the trained model to predict a target values on a test set
        
        For now, we assume that the target value is known, so it is possible to calculate an error value.
        
        Args:
            test_set (torch.utils.data.DataLoader): Data set for which the model predicts a target value. This variable contains batch size information + features + target 
            criterion (torch.nn): Criterion used during training for loss calculation (default = L1Loss() - see: https://pytorch.org/docs/stable/nn.html#loss-functions) 

        Returns:
            mean_loss (float): the average error for all batch of data.
            output (list): Model prediction on the test set
            [inputs, targets] ([list,list]): Group of data containing the input + target of test set
        """
        self._on_predict_start()
        inputs, outputs, targets, test_loss = [],[],[],[]
        for batch_ndx, sample in enumerate(test_set):
            input, target = self.get_batch(sample=sample)
            loss, output = self.loss_calculation(crit, input, target)

            inputs.extend(np.array(input.cpu().detach(), dtype=np.float32))
            targets.extend(np.array(target.cpu().detach(), dtype=np.float32))
            outputs.extend(np.array(output.cpu().detach(), dtype=np.float32))
            test_loss.append(loss.item())
            self.current_batch_test += 1 
            if(self.istrain):
                self.plot_log(epoch=epoch, loss=train_loss, val_loss=np.mean(test_loss), lr=self.opt[0].param_groups[0]['lr'])
            else:
                print(f"Test loss: {loss}")

        mean_loss = np.mean(test_loss)
        self._on_predict_end()
        return mean_loss, np.asarray(outputs), [np.asarray(inputs), np.asarray(targets)]
      
        
    def forward(self, data):
        """Forward propagation.
        
        Note that this function can be overided by subclasses to add specific instructions.
        
        Args:
            data (array of shape (data_number, features_number)): data used for inference.
        
        Returns:
            target (array of shape (data_number, target_number))
        """

        return self.architecture(data)
        
        
    def saveModel(self, path, epoch=None):
        """Save the model state in a json file
        
        If the folder specified does not exist, an error is sent
        If a file already exist, the saved file name is incremented 

        Args:
            path (string): file path without the extension
            epoch (int | None): completed training epoch
        """

        iterator = 1
        while(exists(path + str(iterator) + ".json")):
            iterator+=1

        torch.save(self.architecture.state_dict(), path + "_" + str(iterator) + ".json")
        #print("Saved model to " + path + "_" + str(iterator))
        
        
    def loadModel(self, path):    
        """Load a model from a file

        Args:
            path (string): file path to load without extension
        """
        
        self.architecture.load_state_dict(torch.load(path + ".json", map_location=self.device))
        self.architecture.to(self.device)
        print("Loaded model from disk")
        

    def plotLoss(self, loss_train, loss_val):
        """Plot the loss after training and save it in folder.

        Args:
            loss_train (list of list): loss values collected on train set
            loss_val (list): loss values collected on validation set
        """
        loss_train = loss_train[0]
        fig = plt.figure()
        plt.plot(loss_train, color='blue')
        plt.plot(loss_val, color='red')

        plt.legend(["Training", "Validation"])

        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.grid(True)
        
        # displaying the title
        plt.title("Loss training")
        plt.show()
        fig.savefig("models/tmp/loss_history.png")
        
    def printArchitecture(self, input_shape):
        """Display neural netwotk architecture
        
        Note that the output size of each layer depends on the input shape given to the model (helps to get a good understansing in case of convolution layers)

        Args:
            input_shape (tensor of shape (batch_size, input_size)): Shape of the input normally given to the model.
        """
        
        print("\nNeural network architecture: \n")
        print(f"Input shape: {input_shape}")
        summary(self, input_shape)
        print("\n")





if __name__ == "__main__":
    #!!! TEST SCRIPT !!!
    TEST = "NN"
    if TEST == "NN": 
        import lr_ai.features.preprocessing as pr
        import lr_ai.features.preprocessing as imgpr
        from lr_ai.model.neural_network_architectures.FCNN import *
        from lr_ai.model.neural_network_architectures.CNN import *
        from lr_ai.model.neural_network_architectures.temporal import *        
        from lr_ai.config.NN_config import *
        from lr_ai.config.data_config import *
        
        

        train_path = val_path = test_path = "data/Debug"
        #model.printArchitecture((batch_size,3,224,224) if seq_length == 0 else (batch_size,1,3,224,224))
        
        # !!! Init !!!
        temporal = (seq_length != 0)

        # Use 1 if crop = None, 0 otherwise (segmentation fault in dataloader if other values)
        if crop is not None:
            nb_workers = 0 
        else:
            nb_workers = 1
        nb_workers=0
        
        #!!! Load data set !!!     
        
        if(input_type == InputType.TRACKER):
            train_set, scaler = pr.loader(train_path, shuffle = True, batch_size=batch_size, scaler=scaler, coords=coords, tracker_version=tracker_version, temporal=temporal, sequence_length=seq_length)
            val_set,_ = pr.loader(val_path, shuffle = True, batch_size=batch_size, scaler=scaler, coords=coords, tracker_version=tracker_version, temporal=temporal,sequence_length=seq_length)
            test_set,_ = pr.loader(test_path, shuffle = True, batch_size=batch_size, scaler=scaler, coords=coords, tracker_version=tracker_version, temporal=temporal,sequence_length=seq_length)
        else:
            train_set = imgpr.img_loader(train_path,True,batch_size=batch_size,crop=crop,shuffle=False,temporal=temporal,sequence_length=seq_length)
            val_set = imgpr.img_loader(val_path,False,batch_size=batch_size,crop=crop,shuffle=False,temporal=temporal,sequence_length=seq_length)
            test_set = imgpr.img_loader(test_path,False,batch_size=batch_size,crop=crop,shuffle=False,temporal=temporal,sequence_length=seq_length)
        



        #!!! Training!!! 
        losses_train, losses_val = NN_model.fit(train_set,
        num_epoch, 
        criterion=crit, 
        optimizer=optimizer,
        learning_rate=lr,
        weight_decay=wd, 
        valid_set=val_set)

        NN_model.saveModel(f"models/tmp/NN_epoch{num_epoch}")
        NN_model.plotLoss(losses_train,losses_val)
        

        #!!! Testing !!!
        #model.loadModel('models/LSTM2_1.json')
        score, out = NN_model.predict(test_set)
        print(f"Test loss: {score}")

