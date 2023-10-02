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
        self.outputs = outputs
        self.inputs = inputs
        
        #If available -> work on GPU
        self.device = torch.device('cuda:0' if is_available() else 'cpu')
        print(f"Pytorch is setup on: {self.device}")
        self.architecture = nn.Sequential().to(self.device)

        self.model_name =  "NeuralNetwork"

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
            use_scheduler=False,
            loss_indicators=1):
        """Train a model on a training set
        print(f"Pytorch is setup on: {self.device}")

        
        Args:
            train_set (torch.utils.data.DataLoader): Training set used to fit the model. This variable contains batch size information + features + target 
            epochs (int): Amount of epochs to perform during training
            criterion (torch.nn): Criterion used during training for loss calculation (default = L1Loss() - see: https://pytorch.org/docs/stable/nn.html#loss-functions) 
            optimizer (torch.optim): Optimizer used during training for backpropagation (default = Adam - see: https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
            learning_rate: learning_rate used during backpropagation (default = 0.001)
            valid_set (torch.utils.data.DataLoader): Validation set used to verify model learning. Not mandatory (default = None)
            use_scheduler (bool): Set to true to use a learning rate scheduler (default = False)
            loss_indicators (int): Number of loss indicators used during training. Most NN only need one indicator, but distillation models need three (loss, loss_trainer, loss_student). (default = 1)
        """
        use_gpu=False
        #Use GPU if available
        if torch.cuda.is_available():
            print("CUDA compatible GPU found")
            use_gpu=True
        else:
            print("No CUDA compatible GPU found")
        
        crit = criterion
        if weight_decay is None:
            opt = optimizer(self.architecture.parameters(), lr=learning_rate)
        else:
            opt = optimizer(self.architecture.parameters(), lr=learning_rate, 
                            weight_decay=weight_decay)
        
        scaler = amp.GradScaler(enabled=use_gpu)
        if(use_scheduler):
            scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                            max_lr=learning_rate, 
                                                            epochs=epochs, 
                                                            steps_per_epoch= len(train_set),
                                                            pct_start=0.1)
        
        print("\nTraining START")
        all_loss = []
        losses_train = [[] for _ in range(loss_indicators)]
        losses_val = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = [0 for _ in range(loss_indicators)]
            self.architecture.train()
            # zero the parameter gradients
            opt.zero_grad()

            batch_loss = [[] for _ in range(loss_indicators)]
            progress_bar = tqdm(enumerate(train_set), total=len(train_set))
            dataset_size = 0
            for i, data in progress_bar:
                # get the inputs; data is a list of [inputs, target]
                if use_gpu:
                    inputs = data[0].to(self.device, non_blocking=True)
                    target = data[1].to(self.device)
                    self.architecture.to(self.device)
                else:
                    inputs, target = data[0], data[1]
                # forward + backward + optimize

                losses = self.loss_calculation(crit,inputs,target)
                loss = losses[0]
                #print(f"inputs: {inputs}")
                #print(f"targets: {target}")

                # print statistics
                for i in range(loss_indicators):
                    batch_loss[i].append(losses[i].cpu().item())
                    running_loss[i] += losses[i].item() * inputs.size(0)
                    
                display_loss = np.mean(batch_loss[0])
                all_loss.append(loss.cpu().item())
                dataset_size += inputs.size(0)

                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                if(use_scheduler):
                    s = ('%10s' * 1 + 'Loss :%10.4g' * 1 + ' Learning Rate: %10.4g') % ('%g/%g' % (epoch+1, epochs), display_loss, scheduler.get_last_lr()[0])
                    scheduler.step()
                else:
                    s = ('%10s' * 1 + 'Loss :%10.4g' * 1 + ' Learning Rate: %10.4g') % ('%g/%g' % (epoch+1, epochs), display_loss, learning_rate)
                progress_bar.set_description(s)
            mbl = np.mean(np.sqrt(batch_loss)).round(4) 
            
            if epoch == 0: #Print network architecture
                draw_graph(self.architecture, input_data=inputs, save_graph=True,directory="models/tmp/")

            if epoch % 1 == 0:    # print every epoch
                print(f"Epoch [{epoch+1}/{epochs}], Batch loss: {mbl}")

            if epoch % 100 == 0: #Save model every X epochs
                self.saveModel(f"models/tmp/epoch{epoch}")
            
            if valid_set is not None and epoch%1==0: #Calculate validation loss
                loss,_,_ = self.predict(valid_set, crit=criterion)
                losses_val.append(loss)
                print(f"\nValidation error is {loss}")

            epoch_losses = [[] for _ in range(loss_indicators)]
            for i in range(loss_indicators):
                epoch_losses[i] = running_loss[i] / dataset_size
                losses_train[i].append(epoch_losses[i])


        print('Finished Training')
        return losses_train, losses_val


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
        self.architecture.eval()
        progress_bar = tqdm(enumerate(test_set), total=len(test_set))
        use_gpu = torch.cuda.is_available()
        with torch.no_grad():
            inputs, outputs, targets, test_loss = [],[],[],[]
            for i, data in progress_bar:
                # get the inputs; data is a list of [inputs, target]
                features = data[0].to(self.device, non_blocking=True)
                target = data[1]

                # forward
                #with amp.autocast(enabled=use_gpu):
                output = self.forward(features)

                inputs.extend(np.array(features.cpu().detach(), dtype=np.float32))
                targets.extend(np.array(target.cpu().detach(), dtype=np.float32))
                outputs.extend(np.array(output.cpu().detach(), dtype=np.float32))
                loss = crit(output.cpu().float(), target)
                test_loss.append(loss.item())

        mean_loss = np.mean(test_loss)
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
    
    def loss_calculation(self, crit, inputs, target):
    
        outputs = self.forward(inputs)
        loss = crit(outputs, target) 

        return [loss] 
        
        
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
        print("Saved model to " + path + "_" + str(iterator))
        
        
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

