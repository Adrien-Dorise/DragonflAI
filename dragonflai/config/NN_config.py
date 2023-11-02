"""
 Parameters for neural network applications
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: June 2023
 Last updated: Adrien Dorise - November 2023
"""

import torch
import torch.nn.functional as F

from dragonflai.experiment import ParamType
import dragonflai.model.neural_network_architectures.lookAtScreen as classi
import dragonflai.model.neural_network_architectures.CNN as CNN
import dragonflai.model.neural_network_architectures.FCNN as FCNN
import dragonflai.model.neural_network_architectures.temporal as temporal


param = ParamType.MOVE_CURSOR

#Mouse prediction parameters
if(param == ParamType.MOVE_CURSOR):
    classification = False

    input_size = 13*3
    output_size = 2
    NN_model = FCNN.fullyConnectedNN(input_size,output_size)

    batch_size = 64
    num_epoch = 5
    lr = 1e-4
    wd = 1e-4
    optimizer = torch.optim.AdamW
    crit = torch.nn.L1Loss()


#Classification parameters
elif(param == ParamType.LOOK_AT_SCREEN):
    classification = True
    output_size = 2
    
    #classic classification
    NN_model = classi.VGG11_Light(output_size)
    #NN_model.loadModel("models/distraction_detection/VggLight_3subArch20ep") 
    
    #distillation
    '''
    model = classi.VGG11_Light(output_size)
    model.loadModel("models/distraction_detection/VggLight_3subArch20ep")
    NN_model = classi.VGG11_KL_Distillation(output_size,model=model)
    #NN_model.loadModel("models/distraction_detection/Distil_4subArch20ep_1")
    '''
    
    batch_size = 64
    num_epoch = 5
    lr = 1e-4
    wd = 1e-4
    optimizer = torch.optim.AdamW
    crit = classi.CustomCrossEntropyLoss()