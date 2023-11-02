'''
This package references all neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain - Julia Cohen ({adorise, evillain, jcohen}@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Adrien Dorise - August 2023
'''

import lr_ai.features.preprocessing as pr
import lr_ai.features.image_preprocessing as imgpr
import lr_ai.visualisation.display_mouse_prediction as visu
import lr_ai.visualisation.draw_tracker as draw_tracker
#import lr_ai.visualisation.evaluate_tracker as eval
from lr_ai.model.machineLearning import Regressor
from lr_ai.model.neural_network_architectures.lookAtScreen import lookAtScreenDistillation, LookAtScreenClassification

import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
from enum import Enum
import os

class ParamType(Enum):
    MOVE_CURSOR = 1
    LOOK_AT_SCREEN = 2

class InputType(Enum):
    IMAGE = 0
    TRACKER = 1

class ModelType(Enum):
     MACHINE_LEARNING = 1
     NEURAL_NETWORK_CURSOR = 2
     NEURAL_NETWORK_LOOKATSCREEN = 3
     NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION = 4

class LookAtScreenOptions(Enum):
    LIVE_DEFAULT = 1
    LIVE_GRADCAM = 2
    OFFLINE_GRADCAM = 3
    OFFLINE_SALIENCY = 4
    OFFLINE_MEAN_SALIENCY = 5

class Experiment():
    """Regroup both a model and the data for a complete experiment.
    It can take into account both machine learning and neural network models
    It can fit with both tracker and full image datasets. Only the paths are stored, to avoid saving the whole dataset as Python variables. Data are loaded on the fly when needed (during fit or predict).

    Args:
            model (model class (such as Regressor/NeuralNetwork): algorithm selected
            train_path (string): Folder path containing the training samples
            val_path (string): Folder path containing the validation samples
            test_path (string): Folder path containing the testing samples
            visu_path (string): Folder path containing the samples used for result visualisation
            input_type (InputType enum, optional): Type of feature used (ex: tracker, image). Defaults to InputType.IMAGE.
            crop (string, optional): Select the crop used for image input. Choices are {None, "face", "eyes"}. Defaults to None.
            coords (str, optional): Coordinates used for the tracker features. Choices are {"xy", "xyz"}. Defaults to "xyz".
            tracker_version: int, optional
                identifier of the set of keypoints to keep. Defaults to 1. 
                Options are:
                    - 0: keep all 478 points.
                    - 1: V1, keep 13 points (2pts for the sides of the face, 1pt for 
                            the chin, 4pts for the corners of the eyes, 4pts for the 
                            face vertical axis, 2pts for the corners of the mouth)
                    - 2: V2, keep 69 points (2pts for the sides of the face, 1pt for 
                            the chin, 10pts for the pupils, 32pts for the eyes, 4pts 
                            for the face vertical axis, 20 pts for the mouth contour)
            sequence_length (int, optional):  Only used when temporal set to True. Number of previous observation to take in the model. Default to 0
            num_epoch (int, optional): Amount of epochs to perform during training. Defaults to 50.
            batch_size (int, optional): batch_size used for DataLoader. Defaults to 32.
            learning_rate (int, optional): learning_rate used during backpropagation. Defaults to 1e-03.
            weight_decay (int, optional): regularisation criterion. Defaults to 1e-03.
            optimizer (torch.nn, optional): Optimizer used during training for backpropagation. Defaults to torch.optim.Adam.
            criterion (torch.optim, optional): Criterion used during training for loss calculation. Defaults to torch.nn.L1Loss().
            scaler (sklearn.preprocessing, optional): Type of scaler used for data preprocessing. Put None if no scaler needed. Defaults to MinMaxScaler().
            nb_workers (int, optional):  How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default to 0. Defaults to 0.
    """
    def __init__(self, model,
                train_path,
                val_path,
                test_path,
                visu_path,
                classification = False,
                input_type = InputType.IMAGE,
                crop=None,
                coords="xyz",
                tracker_version=0,
                sequence_length=0,
                num_epoch=50,
                batch_size=32,
                learning_rate=1e-03,
                weight_decay=1e-03,    
                optimizer=torch.optim.Adam,
                criterion=torch.nn.L1Loss(),
                scaler = MinMaxScaler(),
                nb_workers=0):
        #Model parameters  
        self.model = model
        self.classification = classification
        if(type(model) == Regressor):
            self.model_type = ModelType.MACHINE_LEARNING
            self.no_batch = True
        elif(issubclass(type(model),LookAtScreenClassification)):
            if(issubclass(type(model),lookAtScreenDistillation)):
                self.model_type = ModelType.NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION
                self.no_batch = False
                self.loss_indicators = 3
            else:
                self.model_type = ModelType.NEURAL_NETWORK_LOOKATSCREEN
                self.no_batch = False
                self.loss_indicators = 1
        else:
            self.model_type = ModelType.NEURAL_NETWORK_CURSOR
            self.no_batch = False
            self.loss_indicators = 1

        #Path parameters
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.visu_path = visu_path
        
        #Features parameters
        self.scaler = scaler
        self.input_type = input_type
        if(input_type == InputType.IMAGE):
            self.crop = crop
        elif(input_type == InputType.TRACKER):
            self.coords = coords
            self.tracker_version = tracker_version
        
        #Temporal parameters
        self.shuffle = True
        self.is_temporal_model = sequence_length > 0
        self.sequence_length = sequence_length
        if(self.is_temporal_model):
            self.shuffle = False
            
        #Training parameters
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.criterion = criterion
        self.nb_workers = nb_workers
         
    def fit(self):
        """Train the model using the data available in the train and validation folder path.
        """
        # !!! Data loading !!!
        if(self.input_type == InputType.TRACKER):
            train_set, self.scaler = pr.loader(self.train_path, no_batch=self.no_batch, batch_size=self.batch_size, scaler=self.scaler, coords=self.coords, num_workers=self.nb_workers, tracker_version=self.tracker_version, temporal=self.is_temporal_model, sequence_length=self.sequence_length)
            val_set,_ = pr.loader(self.val_path, no_batch=self.no_batch, shuffle=False, batch_size=self.batch_size, scaler=self.scaler, coords=self.coords, num_workers=self.nb_workers, tracker_version=self.tracker_version, temporal=self.is_temporal_model,sequence_length=self.sequence_length)
        elif(self.input_type == InputType.IMAGE):
            train_set = imgpr.img_loader(self.train_path,isTrain=True, shuffle=self.shuffle, batch_size=self.batch_size,crop=self.crop, num_workers=self.nb_workers, temporal=self.is_temporal_model,sequence_length=self.sequence_length,classification=self.classification)
            val_set = imgpr.img_loader(self.val_path, isTrain=False, shuffle=False, crop=self.crop, num_workers=self.nb_workers, temporal=self.is_temporal_model,sequence_length=self.sequence_length,classification=self.classification)
        

        #!!! Training!!! 
        if(self.model_type == ModelType.NEURAL_NETWORK_CURSOR or self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN or self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION):
            losses_train, losses_val = self.model.fit(train_set,
            self.num_epoch, 
            criterion=self.criterion, 
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, 
            valid_set=val_set,
            loss_indicators = self.loss_indicators)
            self.model.plotLoss(losses_train,losses_val)

            if(self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN or self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION):
                self.model.confusion_mat(train_set, 'models/tmp/confusion_matrix.png')

        elif(self.model_type == ModelType.MACHINE_LEARNING):
            self.model.fit(train_set)
            result, out, dataset = self.model.predict(val_set)
            print(f"Test loss for <{self.model}> is: {result}")

        self.model.saveModel(f"models/tmp/model")
        self.save(f"models/tmp/experiment")
        

    def predict(self):          
        """Model prediction on the samples available in the test folder path
        """
        # !!! Data loading !!!
        if(self.input_type == InputType.TRACKER):
            test_set, self.scaler = pr.loader(self.test_path, shuffle=False, no_batch=self.no_batch, batch_size=self.batch_size, scaler=self.scaler, coords=self.coords, num_workers=self.nb_workers, tracker_version=self.tracker_version, temporal=self.is_temporal_model, sequence_length=self.sequence_length)
        elif(self.input_type == InputType.IMAGE):
            test_set = imgpr.img_loader(self.test_path, isTrain=False, shuffle=False, batch_size=self.batch_size,crop=self.crop, num_workers=self.nb_workers, temporal=self.is_temporal_model,sequence_length=self.sequence_length,classification=self.classification)

        score, out, dataset = self.model.predict(test_set,self.criterion)
        if(self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN or self.model_type == ModelType.NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION):
            self.model.confusion_mat(test_set)

        print(f"Test loss: {score}")


    def fit_scaler(self, data_path):
        """Fit the scaler on the given samples.
        Usually done on the training set. 
        The scaler is set beforehand and used trhoughout the experiment to avoid inconsistance  in datasets

        Args:
            data_path (string): Folder path containing the fitting samples
        """
        _, self.scaler = pr.loader(data_path, shuffle = False, no_batch=True, batch_size=self.batch_size, scaler=self.scaler, coords=self.coords, num_workers=self.nb_workers, tracker_version=self.tracker_version, temporal=self.is_temporal_model, sequence_length=self.sequence_length)

    
    def visualise_target_prediction(self, save=False, black_background=False):
        """Visualization of mouse position targets vs predictions.

        Args:
            save (bool, opt): Set to True to save the output folder. Default to False
            black_back (bool, opt): False to get the video playing, True for a black background. Cannot be used when saving video. Default to False
        """

        crop = None
        # !!! Data loading !!!
        if(self.input_type == InputType.TRACKER):
            visu_set,_ = pr.loader(self.visu_path, no_batch=self.no_batch, shuffle=False, batch_size=self.batch_size, scaler=self.scaler, coords=self.coords, num_workers=self.nb_workers, tracker_version=self.tracker_version, temporal=self.is_temporal_model,sequence_length=self.sequence_length)
        elif(self.input_type == InputType.IMAGE):
            visu_set = imgpr.img_loader(self.visu_path, isTrain=False, shuffle=False, crop=self.crop, num_workers=self.nb_workers, temporal=self.is_temporal_model,sequence_length=self.sequence_length)
            crop = self.crop

        _, outputs, (inputs, targets) = self.model.predict(visu_set)
        

        if(self.input_type == InputType.IMAGE and not black_background):
            visu_choice = visu.Background.ORIGINAL
            tracker_version = None
        elif(self.input_type == InputType.IMAGE and black_background):
            visu_choice = visu.Background.BLACK
            tracker_version = None
        elif(self.input_type == InputType.TRACKER and not black_background):
            visu_choice = visu.Background.TRACKER_ORIGINAL
            tracker_version = self.tracker_version
        elif(self.input_type == InputType.TRACKER and black_background):
            visu_choice = visu.Background.TRACKER_BLACK
            tracker_version = self.tracker_version

        visu.compare_target_prediction(self.visu_path,targets, outputs, save, visu_choice, tracker_version, crop)    

    def visualise_lookAtScreen(self, option = LookAtScreenOptions.OFFLINE_GRADCAM):
        """Visualisation of the "look at screen" feature. This feature determines if a user is currently looking directly at the screen, based on a classification model.
        The prediction characteristics are shown through a "relevant feature map" chose by the user with <option> parameter.
        This method only works with LookAtScreen models. 

        Args:
            option (LookAtScreenOptions enum, optional): Options given by the user. Default is OFFLINE_GRADCAM.
                Choices are:
                LIVE_DEFAULT: Live capture of the webcam. The predictions are given in real time
                LIVE_GRADCAM: Live capture of the webcam. The images are then processed to give a gradcam display. The predictions are given in real time
                OFFLINE_GRADCAM: Gradcam processing of given images. Use of the pictures given in the "visu_path" folder of the experiment class. Folder must contain images
                OFFLINE_SALIENCY: Saliency map processing of given images. Use of the pictures given in the "visu_path" folder of the experiment class. Folder must contain images
                OFFLINE_MEAN_SALIENCY: Compute the saliency map for each picture in the folder, and show the average for each map.
        """
        
        if(self.model_type != ModelType.NEURAL_NETWORK_LOOKATSCREEN and self.model_type != ModelType.NEURAL_NETWORK_LOOKATSCREEN_DISTILLATION):
            print("WARNING in visulation_look_at_screen: The model contained in the class is not a LookAtScreen classification model. Can't proceed.")
            return -1

        if(option == LookAtScreenOptions.LIVE_DEFAULT):
            self.model.videoCap()
        
        elif(option == LookAtScreenOptions.LIVE_GRADCAM):
            self.model.videoCap_gradCam()

        elif(option == LookAtScreenOptions.OFFLINE_GRADCAM):
            idx=0
            for img in os.listdir(self.visu_path):
                save_path = "output/gradcam" + str(idx) + ".png"
                idx+=1
                self.model.gradCam(self.visu_path + "/" + img, show=False, save_path=save_path)
            
        elif(option == LookAtScreenOptions.OFFLINE_SALIENCY):
            idx=0
            for img in os.listdir(self.visu_path):
                save_path = "output/saliencymap" + str(idx) + ".png"
                idx+=1
                self.model.saliencyMap(self.visu_path + "/" + img, show=False, save_path=save_path)
        
        elif(option == LookAtScreenOptions.OFFLINE_MEAN_SALIENCY):
            self.model.Saliency_map([self.visu_path + "/" + img for img in os.listdir(self.visu_path)])

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