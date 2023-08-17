import torch
import torch.nn as nn
import torchvision

from LR_AI.model.neuralNetwork import NeuralNetwork


class ConvBlock(nn.Module):
    def __init__(self, In_Channels, Out_Channels, Kernel_Size, Stride, Padding):
        super(ConvBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channels=In_Channels, out_channels=Out_Channels, kernel_size=Kernel_Size, stride=Stride, padding=Padding)
        self.Batch_Norm = nn.BatchNorm2d(num_features=Out_Channels)
        self.Activ_Func = nn.ELU(1.0)
    
    """
    Now we'll build the forward function which defines the path to input tensor
    meaning that we tell the tensor the sequence of layers you're going through 
    Takecare the name of forward function is sensitive so you have to name forward not any thing else
    """
    def forward(self, Tensor_Path):
        Tensor_Path = self.Conv(Tensor_Path)
        Tensor_Path = self.Batch_Norm(Tensor_Path)
        Tensor_Path = self.Activ_Func(Tensor_Path)
        
        return Tensor_Path
    
class InceptionBlock(torch.nn.Module):
    def __init__(self,In_Channels, Num_Of_Filters_1x1, Num_Of_Filters_3x3, Num_Of_Filters_5x5, Num_Of_Filters_3x3_Reduce,Num_Of_Filters_5x5_Reduce, Pooling):
        super(InceptionBlock, self).__init__()
        # The In_Channels are the depth of tensor coming from previous layer
        # First block contains only filters with kernel size 1x1
        self.Block_1 = nn.Sequential(ConvBlock(In_Channels=In_Channels, Out_Channels=Num_Of_Filters_1x1, Kernel_Size=(1,1), Stride=(1,1), Padding=(0,0)))
        
        # Second Block contains filters with kernel size 1x1 followed by 3x3
        self.Block_2 = nn.Sequential(
            ConvBlock(In_Channels=In_Channels, Out_Channels= Num_Of_Filters_3x3_Reduce, Kernel_Size=(1,1), Stride=(1,1), Padding=(0,0)),
            ConvBlock(In_Channels=Num_Of_Filters_3x3_Reduce, Out_Channels= Num_Of_Filters_3x3, Kernel_Size=(3,3), Stride=(1,1), Padding=(1,1))
        )
        
        # Third Block same as second block unless we'll replace the 3x3 filter with 5x5 
        self.Block_3 = nn.Sequential(
            ConvBlock(In_Channels=In_Channels, Out_Channels= Num_Of_Filters_5x5_Reduce, Kernel_Size=(1,1), Stride=(1,1), Padding=(0,0)),
            ConvBlock(In_Channels=Num_Of_Filters_5x5_Reduce, Out_Channels= Num_Of_Filters_5x5, Kernel_Size=(5,5), Stride=(1,1), Padding=(2,2))
        )
        
        # Fourth Block contains maxpooling layer followed by 1x1 filter
        self.Block_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            ConvBlock(In_Channels=In_Channels, Out_Channels=Pooling, Kernel_Size=(1,1), Stride=(1,1), Padding=(0,0))
        )
        
    def forward(self, Tensor_Path):
        First_Block_Out = self.Block_1(Tensor_Path)
        Second_Block_Out = self.Block_2(Tensor_Path)
        Third_Block_Out = self.Block_3(Tensor_Path)
        Fourth_Block_Out = self.Block_4(Tensor_Path)
        
        Concatenated_Outputs = torch.cat([First_Block_Out,Second_Block_Out, Third_Block_Out, Fourth_Block_Out], dim=1) #dim=1 because we want to concatenate in the depth dimension
        return Concatenated_Outputs

class inceptionNet(NeuralNetwork):
    def __init__(self, outputs=10):
        NeuralNetwork.__init__(self, outputs)
        
        self.architecture.add_module('conv1', nn.Conv2d(3,8,5))
        nn.init.xavier_uniform_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('BN1', nn.BatchNorm2d(8))
        nn.init.ones_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('elu1', nn.ELU(1.0))
        self.architecture.add_module('conv1_1', nn.Conv2d(8,8,5))
        nn.init.xavier_uniform_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('BN1_1', nn.BatchNorm2d(8))
        nn.init.ones_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('relu1_1', nn.ELU(1.0))
        self.architecture.add_module('maxPool1', nn.MaxPool2d(2,2))
        
        self.architecture.add_module('inception_1', InceptionBlock(In_Channels=8, Num_Of_Filters_1x1=16, Num_Of_Filters_3x3=16
                                          , Num_Of_Filters_5x5=16, Num_Of_Filters_3x3_Reduce=32, 
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=16))
        self.architecture.add_module('maxPool2', nn.MaxPool2d(2,2))
        self.architecture.add_module('inception_2', InceptionBlock(In_Channels=16*4, Num_Of_Filters_1x1=16, Num_Of_Filters_3x3=16
                                          , Num_Of_Filters_5x5=16, Num_Of_Filters_3x3_Reduce=32, 
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=16))
        self.architecture.add_module('maxPool3', nn.MaxPool2d(2,2))
        self.architecture.add_module('inception_3', InceptionBlock(In_Channels=16*4, Num_Of_Filters_1x1=16, Num_Of_Filters_3x3=16
                                          , Num_Of_Filters_5x5=16, Num_Of_Filters_3x3_Reduce=32, 
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=16))
        self.architecture.add_module('maxPool4', nn.MaxPool2d(2,2))
        self.architecture.add_module('inception_4', InceptionBlock(In_Channels=16*4, Num_Of_Filters_1x1=16, Num_Of_Filters_3x3=16
                                          , Num_Of_Filters_5x5=16, Num_Of_Filters_3x3_Reduce=32, 
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=16))
        self.architecture.add_module('maxPool5', nn.MaxPool2d(2,2))
        self.architecture.add_module('inception_5', InceptionBlock(In_Channels=16*4, Num_Of_Filters_1x1=16, Num_Of_Filters_3x3=16
                                          , Num_Of_Filters_5x5=16, Num_Of_Filters_3x3_Reduce=32, 
                                           Num_Of_Filters_5x5_Reduce=32, Pooling=16))
        self.architecture.add_module('maxPool6', nn.MaxPool2d(2,2))
       
        self.architecture.add_module('flat2', nn.Flatten())
        self.architecture.add_module('drop0', nn.Dropout(p=0.4))

        self.architecture.add_module('lin3', nn.Linear(64*3*3, 512))
        nn.init.xavier_uniform_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('BN6', nn.BatchNorm1d(512))
        nn.init.ones_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('elu6', nn.ELU(1.0))
        self.architecture.add_module('drop1', nn.Dropout(p=0.4))

        self.architecture.add_module('lin5', nn.Linear(512, 256))
        nn.init.xavier_uniform_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('BN8', nn.BatchNorm1d(256))
        nn.init.ones_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)
        self.architecture.add_module('elu8', nn.ELU(1.0))
        #self.architecture.add_module('drop3', nn.Dropout(p=0.3))

        self.architecture.add_module('lin6', nn.Linear(256, self.outputs))
        nn.init.xavier_uniform_(self.architecture[-1].weight.data)
        nn.init.zeros_(self.architecture[-1].bias.data)

        self.architecture.add_module('sigmoid', nn.Sigmoid())
                
class convolutionalNN(NeuralNetwork):
    """Example of a convolutional neural network model.
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, outputs=10):
        NeuralNetwork.__init__(self, outputs)
        
        #Model construction
        #To USER: Adjust your model here
        
        self.architecture.add_module('conv1', nn.Conv2d(3,6,5))
        self.architecture.add_module('relu1', nn.ReLU())
        self.architecture.add_module('maxPool1', nn.MaxPool2d(2,2))
        
        self.architecture.add_module('conv2', nn.Conv2d(6,16,5))
        self.architecture.add_module('relu2', nn.ReLU())
        self.architecture.add_module('maxPool2', nn.MaxPool2d(2,2))
        self.architecture.add_module('flat2', nn.Flatten())
        
        self.architecture.add_module('lin3', nn.Linear(16*5*5, 120))
        self.architecture.add_module('relu3', nn.ReLU())
        
        self.architecture.add_module('lin4', nn.Linear(120, 84))
        self.architecture.add_module('relu4', nn.ReLU())

        self.architecture.add_module('lin5', nn.Linear(84, self.outputs))


class CNN_eyes_tracker(NeuralNetwork):
    """Example of a convolutional neural network model.
    Subclass of NeuralNetwork, the architecture have to be set by the user in the __init__() function.

    """
    def __init__(self, outputs=10):
        NeuralNetwork.__init__(self, outputs)

        self.architecture.add_module('conv1', nn.Conv2d(3,8,5))
        self.architecture.add_module('BN1', nn.BatchNorm2d(8))
        self.architecture.add_module('elu1', nn.ELU(0.1))
        self.architecture.add_module('conv1_1', nn.Conv2d(8,8,5))
        self.architecture.add_module('BN1_1', nn.BatchNorm2d(8))
        self.architecture.add_module('relu1_1', nn.ELU(0.1))
        self.architecture.add_module('maxPool1', nn.MaxPool2d(2,2))

        self.architecture.add_module('conv2', nn.Conv2d(8,8,5))
        self.architecture.add_module('BN2', nn.BatchNorm2d(8))
        self.architecture.add_module('elu2', nn.ELU(0.1))
        self.architecture.add_module('conv2_2', nn.Conv2d(8,16,5))
        self.architecture.add_module('BN2_2', nn.BatchNorm2d(16))
        self.architecture.add_module('elu2_2', nn.ELU(0.1))
        self.architecture.add_module('maxPool2', nn.MaxPool2d(2,2))

        self.architecture.add_module('conv3', nn.Conv2d(16,16,5))
        self.architecture.add_module('BN3', nn.BatchNorm2d(16))
        self.architecture.add_module('elu3', nn.ELU(0.1))
        self.architecture.add_module('conv3_3', nn.Conv2d(16,32,5))
        self.architecture.add_module('BN3_3', nn.BatchNorm2d(32))
        self.architecture.add_module('elu3_3', nn.ELU(0.1))
        self.architecture.add_module('maxPool3', nn.MaxPool2d(2,2))

        self.architecture.add_module('conv4', nn.Conv2d(32,32,5))
        self.architecture.add_module('BN4', nn.BatchNorm2d(32))
        self.architecture.add_module('elu4', nn.ELU(0.1))
        self.architecture.add_module('conv4_4', nn.Conv2d(32,64,5))
        self.architecture.add_module('BN4_4', nn.BatchNorm2d(64))
        self.architecture.add_module('elu4_4', nn.ELU(0.1))
        self.architecture.add_module('maxPool4', nn.MaxPool2d(2,2))


        self.architecture.add_module('conv5', nn.Conv2d(64,128,5))
        self.architecture.add_module('BN5', nn.BatchNorm2d(128))
        self.architecture.add_module('elu5', nn.ELU(0.1))

        self.architecture.add_module('flat2', nn.Flatten())

        self.architecture.add_module('lin3', nn.Linear(128*2*2, 512))
        self.architecture.add_module('BN6', nn.BatchNorm1d(512))
        self.architecture.add_module('elu6', nn.ELU(0.1))
        self.architecture.add_module('drop1', nn.Dropout(p=0.3))

        self.architecture.add_module('lin4', nn.Linear(512, 512))
        self.architecture.add_module('BN7', nn.BatchNorm1d(512))
        self.architecture.add_module('elu7', nn.ELU(0.1))
        self.architecture.add_module('drop2', nn.Dropout(p=0.3))

        self.architecture.add_module('lin5', nn.Linear(512, 512))
        self.architecture.add_module('BN8', nn.BatchNorm1d(512))
        self.architecture.add_module('elu8', nn.ELU(0.1))
        self.architecture.add_module('drop3', nn.Dropout(p=0.3))

        self.architecture.add_module('lin6', nn.Linear(512, self.outputs))


class FaceTrackerNN(NeuralNetwork):
    def __init__(self, outputs=2, backbone="Resnet18", pretrained=True, model_name=None):
        NeuralNetwork.__init__(self, outputs)
        self.backbone, num_channels= self.select_backbone(backbone, pretrained)
        self.head = PredictionHead(num_channels, self.outputs)

        # for print purposes
        self.architecture.append(self.backbone)
        self.architecture.append(self.head)

        self.architecture.to(self.device)

        self.model_name = "FaceTrackerNN" if model_name is None else model_name
    
    def forward(self, data):
        """
        Forward propagation.        
        Parameters
        ----------
        data: torch tensor of input image (N,3,H,W)
            N: Batch size
            W: image width
            H: image Height
        
        Returns
        ----------
        out:
            Mouse cursor predicted normalized position [N, self.outputs] 
        """
        
        out=self.architecture[0](data)
        out=self.architecture[1](out.flatten(1))
        return out

    def select_backbone(self, backbone, pretrained):
        '''
        Select backbone.
        Parameters
        ----------
        backbone : String
            Backbone architecture to be used. Options are: "Resnet18", "Resnet50", "Resnet101", 
            "Resnet34", "EfficientNet" or "CustomModel". 
            "CustomModel" corresponds to the first half of a ResNet18 model.
        pretrained : bool
            Using pretrained model or not.
        Returns
        -------
        backbone : nn.Module
            Returns backbone network.
        num_channels : int
            Number of channels of the last layer.
        '''
        weights = "DEFAULT" if pretrained else None
        if backbone=="Resnet18":
            back = torchvision.models.resnet18(weights=weights)
        elif backbone=="Resnet50":
            back = torchvision.models.resnet50(weights=weights)
        elif backbone=="Resnet101":
            back = torchvision.models.resnet101(weights=weights)
        elif backbone=="Resnet34":
            back = torchvision.models.resnet34(weights=weights)
        elif backbone== "EfficientNet":
            back = torchvision.models.efficientnet_b0(weights=weights)
        elif backbone =="CustomModel":
            back = torchvision.models.resnet18(weights=weights)
            modules = list(back.children())[:-5] #Remove half of the blocks
            num_channels =  list(modules[-1][-1].children())[-2].weight.shape[0]
            backbone_model = nn.Sequential(*modules,nn.AdaptiveAvgPool2d(output_size=(1,1)))
        else:
            print(f"Unknown network {backbone}, exiting...")
            exit()
        if backbone!="CustomModel":
            modules = list(back.children())[:-1] #Remove last layer
            backbone_model = nn.Sequential(*modules,)
            num_channels =  list(modules[-2][-1].children())[-2].weight.shape[0]    #Get # of channels at the end of the backbone
        return backbone_model,num_channels


class PredictionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        Initialise predictor PredictionHead class 
        Parameters
        ----------
        in_channels : int
            Numbre of input channels.

        Returns
        -------
        None.

        '''
        super(PredictionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels/2))
        self.fc2 = nn.Linear(int(in_channels/2), in_channels,)
        self.fc3 = nn.Linear(in_channels, out_channels)
        
    def forward(self, input):
        '''
        Parameters
        ----------
        input : Torch tensor
            input tensor from backbone.

        Returns
        -------
        out : Torch tensor
            Final prediction.

        '''
        out= self.fc3(self.fc2(self.fc1(input))).sigmoid()
        return out