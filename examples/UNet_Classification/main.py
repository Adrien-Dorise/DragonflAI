'''
 Main file to launch a debug test of the LR Technologies artificial intelligence template
 Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
 Created: Avril 2024
 Last updated: Adrien Dorise - Avril 2024
'''

from experiment import *

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as T

import dragonflai.model.neural_network_architectures.UNet as UNet
from dragonflai.model.utils import * 

class OxfordIIITPetsAugmented(datasets.OxfordIIITPet):
    def __init__(self, root: str, split: str, target_types=("segmentation", "category"), download=False, image_size=64):
        super().__init__(root=root, split=split, target_types=target_types, download=download)
        self.image_size = image_size

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        (input, target) = super().__getitem__(idx)

        trans = T.Compose([
            T.ToTensor(),
            T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST),
        ])

        seg, label = target[0], target[1]
        input = trans(input)
        seg = trans(seg)

        seg = seg * 255
        seg = seg.to(torch.long)
        seg -= 1
        seg = seg.squeeze()

        return input, [label, seg]

class CustomLoss(nn.Module):
    def __init__(self, loss4Seg = nn.CrossEntropyLoss(), loss4Classif = nn.CrossEntropyLoss()):
        super(CustomLoss, self).__init__()
        self.loss4Seg = loss4Seg
        self.loss4Classif = loss4Classif

    def forward(self, outputs, targets):

        Lseg = self.loss4Seg(outputs[0], targets[1])
        Lclassif = self.loss4Classif(outputs[1], targets[0])

        return Lseg + Lclassif


if __name__ == "__main__":
    # parameters 
    n_channels              = 3
    image_size              = 32
    n_classes               = 3
    batch_size              = 4
    nb_workers              = 0
    num_epoch               = 10
    lr                      = 1e-3
    wd                      = 1e-4
    optimizer               = torch.optim.Adam
    # crit                    = nn.CrossEntropyLoss()
    crit = CustomLoss()
    scheduler               = torch.optim.lr_scheduler.ReduceLROnPlateau
    numberOfImagesToDisplay = 5

    kwargs_optimizer = {'weight_decay': wd}
    kwargs_scheduler = {'mode': 'min', 'factor': 0.33, 'patience': 1}
    kwargs = {'kwargs_scheduler': kwargs_scheduler}

    base_path = './examples/UNet_Classification'

    input_shape = (batch_size, n_channels, image_size, image_size)
    
    train = OxfordIIITPetsAugmented(base_path + '/data', split="trainval", target_types=("segmentation", "category"), download=True, image_size=image_size)
    test = OxfordIIITPetsAugmented(base_path + '/data', split="test", target_types=("segmentation", "category"), download=True, image_size=image_size)

    train = torch.utils.data.Subset(train, range(50))
    test = torch.utils.data.Subset(test, range(10))

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=nb_workers,
                                                drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test,
                                                batch_size=batch_size ,
                                                shuffle=False,
                                                num_workers=nb_workers,
                                                drop_last=False)
 
    NN_model = UNet.UNet_PET4Classif(n_channels, n_classes)

    NN_model.save_path = base_path + '/results'
    NN_model.progressBar.set_custom_cursor('▄︻デ══━一💨', '-', '⁍', ' ', '🎯')

    experiment = Experiment(NN_model, train_loader, test_loader, 
                num_epoch=num_epoch,
                batch_size=batch_size,
                learning_rate=lr,
                weight_decay=wd,    
                optimizer=optimizer, 
                criterion=crit,
                scheduler=scheduler, 
                kwargs=kwargs, 
                nb_workers=nb_workers,
                numberOfImagesToDisplay=numberOfImagesToDisplay)

    experiment.model.printArchitecture(input_shape)
    experiment.fit()
    experiment.visualise()