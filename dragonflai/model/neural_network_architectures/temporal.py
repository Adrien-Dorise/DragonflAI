import torch.nn as nn
import torch
import numpy as np

from dragonflai.model.neuralNetwork import NeuralNetwork



class extract_tensor_from_LSTM(nn.Module):
    """Extraction module used after a LSTM layer.
    LSTM module from torch output a tuples instead of a tensor. The reason is that LSTM gives the output for EACH sequence.
    This class extract the last output of LSTM.
    """
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        
        try:
            # Reshape shape (batch, hidden)
            tensor = tensor[:,-1,:]
        except:
            raise Exception("Dimension error: You probably forgot to include the seq_lenght in the tensor shape. Maybe the temporal option is not set wen processing the data?")
        
        return tensor  



class LSTM(NeuralNetwork):
    """LSTM architecture
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=1024, output_size=2):
        NeuralNetwork.__init__(self,input_size, output_size)
        #Model construction
        #To USER: Adjust your model here
        

        self.architecture.add_module('LSTM1', nn.LSTM(input_size=input_size, hidden_size=512, num_layers=2, batch_first=True, dropout=0.2))
        for param in self.architecture[-1].parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        self.architecture.add_module('tensor_extraction2', extract_tensor_from_LSTM())
        
        self.architecture.add_module('linear3', nn.Linear(512,256))
        self.architecture.add_module('elu3', nn.ReLU())
        self.architecture.add_module('drop3', nn.Dropout(p=0.2))
        
        self.architecture.add_module('linear4', nn.Linear(256,64))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('elu4', nn.ReLU())

        self.architecture.add_module('linear5', nn.Linear(64,self.outputs))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('sigmoid5', nn.Sigmoid())


            

class CNN_LSTM(NeuralNetwork):

    """CNN_LSTM architecture
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size=224*224, output_size=2):
        NeuralNetwork.__init__(self,input_size, output_size)
        #Model construction
        #To USER: Adjust your model here

        architectureCNN = nn.Sequential()
        architectureLSTM = nn.Sequential()

        architectureCNN.add_module('conv1', nn.Conv2d(3,8,5))
        architectureCNN.add_module('BN1', nn.BatchNorm2d(8))
        architectureCNN.add_module('elu1', nn.ReLU())
        architectureCNN.add_module('conv1_1', nn.Conv2d(8,8,5))
        architectureCNN.add_module('BN1_1', nn.BatchNorm2d(8))
        architectureCNN.add_module('relu1_1', nn.ReLU())
        architectureCNN.add_module('maxPool1', nn.MaxPool2d(2,2))

        architectureCNN.add_module('conv2', nn.Conv2d(8,8,5))
        architectureCNN.add_module('BN2', nn.BatchNorm2d(8))
        architectureCNN.add_module('elu2', nn.ReLU())
        architectureCNN.add_module('conv2_2', nn.Conv2d(8,16,5))
        architectureCNN.add_module('BN2_2', nn.BatchNorm2d(16))
        architectureCNN.add_module('elu2_2', nn.ReLU())
        architectureCNN.add_module('maxPool2', nn.MaxPool2d(2,2))

        architectureCNN.add_module('conv3', nn.Conv2d(16,16,5))
        architectureCNN.add_module('BN3', nn.BatchNorm2d(16))
        architectureCNN.add_module('elu3', nn.ReLU())
        architectureCNN.add_module('conv3_3', nn.Conv2d(16,32,5))
        architectureCNN.add_module('BN3_3', nn.BatchNorm2d(32))
        architectureCNN.add_module('elu3_3', nn.ReLU())
        architectureCNN.add_module('maxPool3', nn.MaxPool2d(2,2))

        architectureCNN.add_module('conv4', nn.Conv2d(32,32,5))
        architectureCNN.add_module('BN4', nn.BatchNorm2d(32))
        architectureCNN.add_module('elu4', nn.ReLU())
        architectureCNN.add_module('conv4_4', nn.Conv2d(32,64,5))
        architectureCNN.add_module('BN4_4', nn.BatchNorm2d(64))
        architectureCNN.add_module('elu4_4', nn.ReLU())
        architectureCNN.add_module('maxPool4', nn.MaxPool2d(2,2))

        architectureCNN.add_module('conv5', nn.Conv2d(64,128,5))
        architectureCNN.add_module('BN5', nn.BatchNorm2d(128))
        architectureCNN.add_module('elu5', nn.ReLU())

        architectureCNN.add_module('flat5', nn.Flatten())
        #architectureCNN.add_module('reshapeLSTM5',shape_tensor_for_LSTM())   
        

        architectureLSTM.add_module('LSTM6', nn.LSTM(input_size=128*2*2, hidden_size=512, num_layers=2, batch_first=True, dropout=0.2))
        for param in architectureLSTM[-1].parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
        architectureLSTM.add_module('tensor_extraction6', extract_tensor_from_LSTM())
        
        architectureLSTM.add_module('linear7', nn.Linear(512,256))
        architectureLSTM.add_module('elu7', nn.ReLU())
        architectureLSTM.add_module('drop7', nn.Dropout(p=0.2))

        architectureLSTM.add_module('linear8', nn.Linear(256,128))
        nn.init.xavier_uniform_(architectureLSTM[-1].weight)
        architectureLSTM.add_module('elu8', nn.ReLU())
        architectureLSTM.add_module('drop8', nn.Dropout(p=0.2))

        architectureLSTM.add_module('linear9', nn.Linear(128,self.outputs))
        nn.init.xavier_uniform_(architectureLSTM[-1].weight)


        #self.architecture = torch.nn.Sequential(*architectureCNN.children(), *architectureLSTM.children()).to(self.device)
        self.architecture.add_module("convLayers", architectureCNN.to(self.device))
        self.architecture.add_module("LSTMLayers", architectureLSTM.to(self.device))

    def forward(self, input):
        output_len_CNN = self.architecture[0](input[0:1,0,:,:]).shape[-1]

        outputCNN = torch.zeros((input.shape[0],input.shape[1],output_len_CNN)).to(self.device)
        for seq in range(np.shape(input)[1]):
            outputCNN[:,seq,:] = self.architecture[0](input[:,seq,:,:])

        outputLSTM = self.architecture[1](outputCNN)

        return outputLSTM

