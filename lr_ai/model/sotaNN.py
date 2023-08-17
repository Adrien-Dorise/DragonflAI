"""
This package references all neural network classes used in the application.
Author: Julia Cohen - Edouard Villain ({jcohen, evillain}@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Edouard Villain - March 2023

Extension of NeuralNetwork class for state-of-the-art networks:
- ResNet-18
- ResNet-101
- MobileNet v2
- EfficientNet v2 Small
- CoAtNet (code from https://github.com/chinhsuanwu/coatnet-pytorch)
"""


from torchvision.models import resnet18, ResNet18_Weights, resnet101, ResNet101_Weights, \
            mobilenet_v2, MobileNet_V2_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from torch.nn.functional import one_hot
from torch.nn import Linear

from lr_ai.model.neuralNetwork import NeuralNetwork
from lr_ai.model.coatnet import coatnet_0

class ResNet18NN(NeuralNetwork):
    """
    Integrate the ResNet18 model from Pytorch inside our NeuralNetwork class.
    """
    def __init__(self, outputs=10, pretrained=True):
        super().__init__(outputs)

        weights = None
        self.tf = None
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.tf = weights.transforms()
        self.architecture = resnet18(weights=weights)

        self.update_outputs()
        self.architecture = self.architecture.to(self.device)
    
    def update_outputs(self):
        in_features = self.architecture.fc.in_features
        bias = self.architecture.fc.bias is not None
        new_fc = Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture.fc = new_fc


class ResNet101NN(NeuralNetwork):
    """
    Integrate the ResNet101 model from Pytorch inside our NeuralNetwork class.
    """
    def __init__(self, outputs=10, pretrained=True):
        super().__init__(outputs)

        weights = None
        self.tf = None
        if pretrained:
            weights = ResNet101_Weights.DEFAULT  # or weights='default'
            self.tf = weights.transforms()
        self.architecture = resnet101(weights=weights)

        self.update_outputs()
        self.architecture = self.architecture.to(self.device)

    def update_outputs(self):
        in_features = self.architecture.fc.in_features
        bias = self.architecture.fc.bias is not None
        new_fc = Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture.fc = new_fc


class MobileNetv2NN(NeuralNetwork):
    """
    Integrate the MobileNet V2 model from Pytorch inside our NeuralNetwork class.
    """
    def __init__(self, outputs=10, pretrained=True):
        super().__init__(outputs)

        weights = None
        self.tf = None
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.tf = weights.transforms()
        
        self.architecture = mobilenet_v2(weights=weights)

        self.update_outputs()
        self.architecture = self.architecture.to(self.device)

    def update_outputs(self):
        last_fc = self.architecture.classifier[1]
        in_features = last_fc.in_features
        bias = last_fc.bias is None
        new_fc = Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture.classifier[1] = new_fc

class EfficientNetv2sNN(NeuralNetwork):
    """
    Integrate the ResNet18 model from Pytorch inside our NeuralNetwork class.
    """
    def __init__(self, outputs=10, pretrained=True):
        super().__init__(outputs)

        weights = None
        self.tf = None
        if pretrained:
            weights = EfficientNet_V2_S_Weights.DEFAULT
            self.tf = weights.transforms()
        
        self.architecture = efficientnet_v2_s(weights=weights)
        self.update_outputs()
        self.architecture = self.architecture.to(self.device)
    
    def update_outputs(self):
        last_fc = self.architecture.classifier[1]
        in_features = last_fc.in_features
        bias = last_fc.bias is None
        new_fc = Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture.classifier[1] = new_fc


class CoAtNetNN(NeuralNetwork):
    def __init__(self, outputs=10, pretrained=True):
        super().__init__(outputs)
        self.tf = None
        self.architecture = coatnet_0()

        self.update_outputs()
        self.architecture = self.architecture.to(self.device)
    
    def update_outputs(self):
        in_features = self.architecture.fc.in_features
        bias = self.architecture.fc.bias is not None
        new_fc = Linear(in_features=in_features, out_features=self.outputs, bias=bias)
        self.architecture.fc = new_fc

class OneHot:
    """
    Transform a vector of classes into one-hot vectors (all 0 with a single 1 for thee corresponding class).
    Used for classification neworks, necessary for some loss functions (like MSE).
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.eye = torch.eye(self.num_classes)
    
    def __call__(self, labels):
        return self.eye[labels]
    
if __name__ == "__main__":
    #!!! TEST SCRIPT !!!
    import time

    from torchvision.transforms import Compose, ToTensor, Normalize, Resize
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    import torch

    #!!! Parameters !!!
    batch_size = 12
    epoch = 1

    #!!! Create NN classes !!!
    userNet = MobileNetv2NN(pretrained=True, outputs=10)
    userNet.printArchitecture((3,32,32))
    #print(userNet.architecture)

    train_tf = userNet.tf
    print("Training tf=", train_tf)

    #!!! Load data set !!!
    resize_value = (32, 32)
    transform = Compose(
        [Resize(resize_value),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #if train_tf is not None:
    #    transform = train_tf
    
    target_transform = Compose(
        [OneHot(num_classes=10)]
    )

    trainset = CIFAR10(root='./data', train=True,
                                download=True, transform=transform,
                                target_transform=target_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, num_workers=4)

    testset = CIFAR10(root='./data', train=False,
                                download=True, transform=transform,
                                target_transform=target_transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    #imshow(make_grid(images))
    # print labels
    #print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


    #!!! Training!!!
    t_start = time.time()
    userNet.fit(trainloader, epoch, softmax=True)
    t_end = time.time()
    print(f"Training time for {epoch} epoch of dataset size {len(trainset)}: {t_end-t_start}sec ({(t_end-t_start)/60} min).")


    #!!! Testing !!!
    t_start = time.time()
    score = userNet.predict(testloader)
    print(f"Score: {score}")
    t_end = time.time()
    print(f"Prediction time for test dataset size {len(testset)}: {t_end-t_start}seconds.")

