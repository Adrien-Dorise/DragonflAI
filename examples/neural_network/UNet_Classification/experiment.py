'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: Avril 2024
Last updated: Adrien Dorise - Avril 2024

'''
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn

int_to_category = {
    0: 'abyssinian',
    1: 'american_bulldog',
    2: 'american_pit_bull_terrier',
    3:  'basset_hound',
    4:  'beagle',
    5:  'bengal',
    6:  'birman',
    7:  'bombay',
    8:  'boxer',
    9: 'british_shorthair',
    10: 'chihuahua',
    11: 'egyptian_mau',
    12: 'english_cocker_spaniel',
    13: 'english_setter',
    14: 'german_shorthaired',
    15: 'great_pyrenees',
    16: 'havanese',
    17: 'japanese_chin',
    18: 'keeshond',
    19: 'leonberger',
    20: 'maine_coon',
    21: 'miniature_pinscher',
    22: 'newfoundland',
    23: 'persian',
    24: 'pomeranian',
    25: 'pug',
    26: 'ragdoll',
    27: 'russian_blue',
    28: 'saint_bernard',
    29: 'samoyed',
    30: 'scottish_terrier',
    31: 'shiba_inu',
    32: 'siamese',
    33: 'sphynx',
    34: 'staffordshire_bull_terrier',
    35: 'wheaten_terrier',
    36: 'yorkshire_terrier'
}

class Experiment():
    def __init__(self, model,
                train_loader, test_loader, 
                num_epoch     = 10,
                batch_size    = 32,
                learning_rate = 1e-03,
                weight_decay  = 1e-03,
                optimizer     = torch.optim.Adam,
                scheduler     = None,
                kwargs        = {},
                criterion     = torch.nn.L1Loss(),
                nb_workers    = 0,
                numberOfImagesToDisplay = 5):
        # Model parameters
        self.model                   = model
        self.num_epoch               = num_epoch
        self.batch_size              = batch_size
        self.learning_rate           = learning_rate
        self.weight_decay            = weight_decay
        self.optimizer               = optimizer
        self.scheduler               = scheduler
        self.kwargs                  = kwargs
        self.criterion               = criterion
        self.nb_workers              = nb_workers
        self.train_loader            = train_loader
        self.test_loader             = test_loader
        self.numberOfImagesToDisplay = numberOfImagesToDisplay

    def fit(self):
        """Train the model using the data available in the train and validation folder path.
        """
        self.model._compile(self.train_loader, self.test_loader, 
                            self.criterion, self.learning_rate, 
                            self.optimizer, self.scheduler, 
                            self.batch_size, self.num_epoch, **self.kwargs)
        
        history = self.model.fit(self.train_loader,
        self.num_epoch, 
        criterion=self.criterion, 
        optimizer=self.optimizer,
        learning_rate=self.learning_rate,
        weight_decay=self.weight_decay, 
        valid_set=self.test_loader,
        loss_indicators=1, 
        batch_size=self.batch_size)
        self.model.plot_learning_curve(history.loss_train,history.loss_val, 'loss')
        self.model.plot_learning_curve(history.acc_train,history.acc_val, 'accuracy')
        self.model.plot_learning_rate(history.lr, 'lr')

    def predict(self):          
        """Model prediction on the samples available in the test folder path
        """
        # !!! Data loading !!!
        _, _, _ = self.model.predict(self.test_loader,self.criterion)

    def visualise(self):
        """Visualisation of the first picture of the test set.
        The input + predicted images are both shown.
        """
        self.confusionmatrixCreator()
        self.numberImagesToDisplay = 1
        self.model.history.verbosity = 0
        _, output, (input, target) = self.model.predict(itertools.islice(self.test_loader, self.numberImagesToDisplay) ,self.criterion)

        seg_gt, class_gt = target[1], target[0]

        seg_prediction = [] # Preparation of segmentation prediction

        for i in range(len(output[0])):
            pred_seg = nn.Softmax(dim=0)(torch.from_numpy(output[0][i])).argmax(dim=0).to(torch.float)
            seg_prediction.append(pred_seg)

        label_prediction = [] # Preparation of label prediction

        for i in range(len(output[1])):
            pred_label = nn.Softmax(dim=0)(torch.from_numpy(output[1][i])).argmax(dim=0).to(torch.float)
            label_prediction.append(pred_label)


        all_imgs = []
        # Fetch test data
        for batch in self.test_loader:
            img, _ = batch
            all_imgs.extend(img)

        target = target.transpose(1,0)
        for _, (inp, targ, seg_predic, label_predic) in \
            enumerate(zip(all_imgs[:self.numberImagesToDisplay], \
                        target[:self.numberImagesToDisplay], \
                        seg_prediction[:self.numberImagesToDisplay], \
                        label_prediction[:self.numberImagesToDisplay])):

            _, axes = plt.subplots(1, 3)

            def convert_tensor2opencv(image):
                image = image.squeeze() * 127
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                img = image.astype(np.uint8)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            inp_cv = cv2.cvtColor(np.transpose((inp.numpy()*255).astype(np.uint8), (1,2,0)), cv2.COLOR_RGB2BGR)

            targ_cv = cv2.applyColorMap(convert_tensor2opencv(targ[1]), cv2.COLORMAP_JET)
            predic_cv = cv2.applyColorMap(convert_tensor2opencv(seg_predic), cv2.COLORMAP_JET)

            overlay_targ = cv2.addWeighted(inp_cv, 0.5, targ_cv, 0.5, 0)  # Adjust the weights as needed
            overlay_pred = cv2.addWeighted(inp_cv, 0.5, predic_cv, 0.5, 0)

            axes[0].imshow(cv2.cvtColor(inp_cv, cv2.COLOR_BGR2RGB))
            axes[0].set_title('RGB Image')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(overlay_targ, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'RGB Image + Target: {int_to_category.get(targ[0])}')
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(overlay_pred, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f'RGB Image + Prediction : {int_to_category.get(int(label_predic))}')
            axes[2].axis('off')

            plt.show()


    def confusionmatrixCreator(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        _, output, (input, target) = self.model.predict(self.test_loader, self.criterion)
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        label_prediction = [] # Preparation of label prediction

        for i in range(len(output[1])):
            pred_label = int(nn.Softmax(dim=0)(torch.from_numpy(output[1][i])).argmax(dim=0).to(torch.float))
            label_prediction.append(pred_label)

        cm = confusion_matrix(target[0].astype('int'), label_prediction)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot().figure_.savefig('{}/CM.png'.format(self.model.save_path))


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
