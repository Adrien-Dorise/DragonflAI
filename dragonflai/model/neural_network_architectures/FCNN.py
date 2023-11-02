import torch.nn as nn

from dragonflai.model.neuralNetwork import NeuralNetwork



class fullyConnectedNN(NeuralNetwork):
    """Example of a fully connected neural network model.
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, input_size, outputs=2):
        NeuralNetwork.__init__(self, input_size, outputs)
        #Model construction
        #To USER: Adjust your model here

        self.architecture.add_module('lin1', nn.Linear(input_size, 512))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu1', nn.ReLU())
        self.architecture.add_module('dropout1', nn.Dropout(p=0.3))
        
        self.architecture.add_module('lin2', nn.Linear(512 , 128))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu2', nn.ReLU())
        self.architecture.add_module('dropout2', nn.Dropout(p=0.2))
        '''
        
        self.architecture.add_module('lin3', nn.Linear(256, 128))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu3', nn.ReLU())
        self.architecture.add_module('dropout3', nn.Dropout(p=0.3))

        self.architecture.add_module('lin4', nn.Linear(256, 128))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu4', nn.ReLU())
        #self.architecture.add_module('dropout4', nn.Dropout(p=0.3))
        '''
        
        self.architecture.add_module('lin3', nn.Linear(128, self.outputs))
        nn.init.xavier_normal_(self.architecture[-1].weight)
        self.architecture.add_module('relu3', nn.Sigmoid())
        
        self.architecture.to(self.device)