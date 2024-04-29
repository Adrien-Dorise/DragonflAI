"""
This package references all neural network classes used in the application.
Author: Julia Cohen - Adrien Dorise - Edouard Villain ({jcohen, adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Edouard Villain - April 2024 

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
from torchinfo import summary
from torchview import draw_graph

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import time 

from dragonflai.model.utils import *


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
    
    def __init__(self, modelType, taskType, name='NeuralNetwork'):
        super().__init__()
        self.use_gpu = torch.cuda.is_available()
        self.save_path   = './results'
        self.history     = dragonflAI_History(modelType, taskType)
        self.progressBar = dragonflAI_ProgressBar(self.history)
        #If available -> work on GPU
        self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        print(f"Pytorch is setup on: {self.device}")
        
        self.architecture = nn.Sequential().to(self.device)
        self.model_name =  name
        
        
    def _compile(self, train_loader, test_loader, crit, lr, opts, scheduler, batch_size, epochs, **kwargs):
        for _, sample in enumerate(train_loader):
            # get batch 
            inputs, targets = self.get_batch(sample=sample)
            _, output = self.loss_calculation(crit, inputs, targets, with_grad=False)
            self.input_shape = inputs.shape
            output_shape = output.shape
            break
        self.opt       = []
        self.scheduler = []
        self.scaler    = []
        self.scaler.append(torch.cuda.amp.GradScaler(enabled=self.use_gpu))
        try:
            kwargs_optimizer = kwargs['kwargs_optimizer']
            self.opt.append(opts(self.architecture.parameters(), lr=lr, **kwargs_optimizer))
        except:
            self.opt.append(opts(self.architecture.parameters(), lr=lr))
        try:
            kwargs_scheduler = kwargs['kwargs_scheduler']
            self.scheduler.append(scheduler(self.opt[0], **kwargs_scheduler))
        except:
            pass 
            
        self.printArchitecture(self.input_shape)
        
        self.init_results(train_loader, test_loader, batch_size, epochs)
        print('Training model {} during {} epochs with batch size set to {} on {} training data and validating on {} data'
              .format(self.model_name, self.history.parameters['nb_max_epochs'], 
                      self.history.parameters['batch_size'], 
                      self.history.parameters['batch_size'] * self.history.parameters['steps_per_epoch_train'], 
                      self.history.parameters['batch_size'] * self.history.parameters['steps_per_epoch_val'],
                      ))
        print('\ninput_shape {}  ====> {} {} ====> output_shape {}\n'.format(
            self.input_shape, self.history.modelType, self.history.taskType, output_shape
            ))
    
    

    def init_results(self, train_loader, test_loader, batch_size, epochs, *args, **kwargs):
        #Use GPU if available
        if self.use_gpu:
            print("CUDA compatible GPU found")
        else:
            print("No CUDA compatible GPU found")
        
        parameters = {
            'nb_max_epochs'        : epochs,
            'batch_size'           : batch_size,
            'dataset_size'         : len(train_loader) * batch_size,
            'steps_per_epoch_train': len(train_loader),
            'steps_per_epoch_val'  : len(test_loader),
            }
        self.history.set_new_parameters(parameters)
        
    def update_scheduler(self, *args, **kwargs):
        '''update scheduler'''
        loss = kwargs['loss']
        if(self.scheduler[0] is not None):
            for scheduler in self.scheduler:
                scheduler.step(loss)

    def get_batch(self, *args, **kwargs):
        '''get batch'''
        sample = kwargs['sample']
        return sample[0].to(self.device), sample[1].to(self.device)
            
    def loss_calculation(self, crit, inputs, target, *args, **kwargs):
        '''compute loss'''
        # get with_grad parameter 
        with_grad=kwargs['with_grad']
        # forward pass with gradient computing 
        if with_grad:
            outputs = self.forward(inputs)
            loss    = crit(outputs, target)
        else: # forward pass without gradient 
            with torch.no_grad():
                outputs = self.forward(inputs)
                loss    = crit(outputs, target)
                torch.cuda.empty_cache()
        
        return loss, outputs

    def train_batch(self, *args, **kwargs):
        '''train a batch '''
        loss = kwargs['loss']
        
        #See here for detail about multiple scaler & optimizer
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
        
        # scaler scaling 
        # retain graph set to true in order to get a complete graph 
        # connecting input to output, even in multi modal cases 
        for idx, scaler in enumerate(self.scaler):
            retain_graph = (idx < len(self.scaler)-1)    
            scaler.scale(loss).backward(retain_graph=retain_graph)    
        
        # step optimizer with its scaler 
        for i in range(len(self.scaler)):
            self.scaler[i].step(self.opt[i])

        # update scaler 
        for scaler in self.scaler:
            scaler.update()
        
        # reset optimizers's gradients  
        for opt in self.opt:
            opt.zero_grad()

    def save_epoch_end(self, *args, **kwargs):   
        if self.history.current_status['current_epoch'] % 100 == 0: #Save model every X epochs
            self.saveModel(f"{self.save_path}/epoch{self.history.current_status['current_epoch']}")
            
        try:  
            if self.history.loss_train[-1] == np.min(self.history.loss_train):
                self.saveModel("{}/{}_best_train".format(self.save_path, self.model_name))
            if self.history.loss_val[-1] == np.min(self.history.loss_val):
                self.saveModel("{}/{}_best_val".format(self.save_path, self.model_name))
        except:
            pass 
        
    
    def fit(self, train_set, epochs, 
            criterion       = nn.L1Loss(),
            optimizer       = torch.optim.Adam,
            learning_rate   = 0.001,
            weight_decay    = None,
            valid_set       = None,
            loss_indicators = 1,
            batch_size      = 2
            ):
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
        # training starting callback 
        self._on_training_start()
        
        # iterate throw epochs 
        for _ in range(epochs):
            # starting epoch callback 
            self._on_epoch_start()
            # iterate throw batch 
            for _, sample in enumerate(train_set):
                # get batch 
                inputs, targets = self.get_batch(sample=sample)
                # forward batch with training 
                _, _ = self.forward_batch(inputs, targets, criterion, train=True)
            # update history 
            self.history._end_train_epoch(loss=np.mean(self.batch_loss), 
                                                lr=self.opt[0].param_groups[0]['lr'], 
                                                acc=self.acc)
            # predict on validation set 
            val_loss, _, _ = self.predict(valid_set, crit=criterion)
            # update sheduler on validation set 
            self.update_scheduler(loss=val_loss)
            # ending epoch callback 
            self._on_epoch_end()
        # ending training callback 
        self._on_training_end()
        self.istrain = False
        return self.history


    def predict(self, test_set, crit=nn.L1Loss()):
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
        # predict starting callback 
        self._on_predict_start()
        # init list 
        self.inputs, self.outputs, self.targets, self.test_loss = [],[],[],[]
        # iterate validation set 
        for _, sample in enumerate(test_set):
            # get batch 
            input, target = self.get_batch(sample=sample)
            # forward batch without training 
            _, _ = self.forward_batch(input, target, crit, train=False)
        # predict ending callback 
        self._on_predict_end()
        
        return np.mean(self.test_loss), np.asarray(self.outputs), [np.asarray(self.inputs), np.asarray(self.targets)]


    def forward_batch(self, input, target, crit, train):
        # start batch callback 
        self._on_batch_start()
        # forward pass and get loss 
        loss, output = self.loss_calculation(crit, input, target, with_grad=train)
        # training mode 
        if train:
            # at first batch of first epoch : draw model in png 
            if self.history.current_status['current_epoch'] == 1 and \
                self.history.current_status['current_batch_train'] == 0: #Print network architecture
                draw_graph(self, input_data=input, save_graph=True, directory=self.save_path, expand_nested=True, depth=5)
            # add current batch loss 
            self.batch_loss.append(loss.cpu().item())
            # backward pass 
            self.train_batch(loss=loss)
            # update accuracy if needed 
            if self.history.taskType == taskType.CLASSIFICATION:
                self._update_acc(output, target)
            # update history 
            self.history._end_train_batch(lr=self.opt[0].param_groups[0]['lr'], 
                                            current_loss_train=np.mean(self.batch_loss),
                                            current_acc_train=self.acc)
        else: # validating mode 
            # add current test loss 
            self.test_loss.append(loss.item()) 
            # get input, target, ouput as array 
            self.inputs.extend(np.array(target.cpu().detach(), dtype=np.float32))
            self.targets.extend(np.array(target.cpu().detach(), dtype=np.float32))
            self.outputs.extend(np.array(output.cpu().detach(), dtype=np.float32))
            # update accuracy if needed 
            if self.history.taskType == taskType.CLASSIFICATION:
                self._update_acc(output, target, val=True)
            # update history 
            self.history._end_val_batch(current_loss_val=np.mean(self.test_loss), current_acc_val=self.acc_val)
        # ending batch callback 
        self._on_batch_end()
                    
        return loss, output 
        
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

        #Check if folder exists
        file_name = path.split("/")[-1]
        folder_path = path[0:-len(file_name)]
        if not exists(folder_path):
            os.makedirs(folder_path)
        
        #Check if file exists
        iterator = 1
        while(exists(path + str(iterator) + ".json")):
            iterator+=1

        torch.save(self.architecture.state_dict(), path + "_" + str(iterator) + ".json")
        
        
    def loadModel(self, path):    
        """Load a model from a file

        Args:
            path (string): file path to load without extension
        """
        try:
            self.architecture.load_state_dict(torch.load(path + ".json", map_location=self.device))
            self.architecture.to(self.device)
            print("Loaded model from disk")
        except Exception:
            raise Exception(f"Error when loading Neural Network model: {path} not found")
        

    def plot_learning_curve(self, train, val, name):
        """Plot the loss after training and save it in folder.

        Args:
            loss_train (list of list): loss values collected on train set
            loss_val (list): loss values collected on validation set
        """
        fig = plt.figure()
        plt.plot(train, color='blue')
        plt.plot(val, color='red')

        plt.legend(["Training", "Validation"])

        plt.xlabel('epoch')
        plt.ylabel(name)

        plt.grid(True)
        
        # displaying the title
        plt.title("{} training".format(name))
        #plt.show()
        fig.savefig("{}/{}_history.png".format(self.save_path, name))
        
        
    def plot_learning_rate(self, lr, name):
        """Plot the loss after training and save it in folder.

        Args:
            loss_train (list of list): loss values collected on train set
            loss_val (list): loss values collected on validation set
        """
        fig = plt.figure()
        plt.plot(lr, color='blue')

        plt.legend(["Learning rate"])

        plt.xlabel('epoch')
        plt.ylabel(name)

        plt.grid(True)
        
        # displaying the title
        plt.title("{} training".format(name))
        #plt.show()
        fig.savefig("{}/{}_history.png".format(self.save_path, name))
        
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
        
        
        
        
    ########### Callback methods        
    def _update_acc(self, output, target, val=False):
        classifications     = torch.argmax(output, dim=1)
        correct_predictions = sum(classifications==target).item()
        if val:
            self.total_correct_val   += correct_predictions
            self.total_instances_val += self.history.parameters['batch_size']
            self.acc_val              = (self.total_correct_val/self.total_instances_val) * 100
        else:
            self.total_correct   += correct_predictions
            self.total_instances += self.history.parameters['batch_size']
            self.acc              = (self.total_correct/self.total_instances) * 100
                
    def _on_batch_end_time(self, *args, **kwargs):
        '''callback function, called at each batch's end'''
        curent_duration_t = time.time() - self.history.current_status['start_epoch_t']
        if self.history.current_status['current_batch_val'] == \
            self.history.parameters['steps_per_epoch_val']:
            self.history.set_current_status('duration_t', np.around(curent_duration_t, decimals=2))
        else:
            nb_batch_done   = self.history.current_status['current_batch_train'] + self.history.current_status['current_batch_val']
            total           = self.history.parameters['steps_per_epoch_train'] + self.history.parameters['steps_per_epoch_val']
            ratio           = nb_batch_done / total
            est             = curent_duration_t / ratio
            self.history.set_current_status('duration_t', np.around(est - curent_duration_t, decimals=2))
        
    def _on_epoch_start(self, *args, **kwargs):
        '''callback function, called at each epoch's start'''
        self.architecture.train() 
        self.batch_loss          = []
        self.total_correct       = 0
        self.total_instances     = 0
        self.acc                 = 0
        self.total_correct_val   = 0
        self.total_instances_val = 0
        self.acc_val             = 0
        self.history._start_epoch()
        self.progressBar.plot_log()
        
    def _on_predict_start(self, *args, **kwargs):
        '''callback function, called at each predict start'''
        self.architecture.eval()
        self.history.set_current_status('current_batch_test', 0)
        
    def _on_predict_end(self, *args, **kwargs):
        '''callback function, called at each predict end'''
        self.history._end_val_epoch(loss=np.mean(self.test_loss), acc=self.acc_val) 
    
    def _on_epoch_end(self, *args, **kwargs):
        '''callback function, called at each epoch's end'''
        self.save_epoch_end() 
    
    def _on_batch_start(self, *args, **kwargs):
        '''callback function, called at each batch's start'''
        pass 
    
    def _on_batch_end(self, *args, **kwargs):
        '''callback function, called at each batch's end'''
        self._on_batch_end_time()
        self.progressBar.plot_log()
    
    def _on_training_start(self, *args, **kwargs):
        '''callback function, called at training start'''
        print('\tStart training...') 
    
    def _on_training_end(self, *args, **kwargs):
        '''callback function, called at training end'''
        print('\tEnd training...')
