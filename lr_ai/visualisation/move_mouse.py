"""
This package references all neural network classes used in the application.
Author: Julia Cohen (jcohen@lrtechnologies.fr) - LR Technologies
Created: March 2023
Last updated: Adrien Dorise (adorise@lrtechnologies.fr) - August 2023
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
import pyautogui

from LR_AI.model.neuralNetwork import FaceTrackerNN
from LR_AI.model.machineLearning import Regressor
from LR_AI.features.preprocessing import load_metadata, getTrackerSet
from LR_AI.visualisation.display_mouse_prediction import draw_point_on_frame
from LR_AI.features.tracker_toolbox import crop_face, get_landmarks, get_detector

im2ten_tf = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

def image2batch(image, shape, detector=None):
    """
    Convert a BGR cv2 image into a Pytorch batch of 1 tensor.

    Args:
        image (np.ndarray): BGR frame from cv2 video
        shape (tuple): tuple of (w, h) to resize the input image

    Returns:
        torch.Tensor: float tensor of shape (1, img_height, img_width, 3)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if detector is not None:
        landmarks = get_landmarks(image, detector)
        if landmarks is None: 
            print("No face detected in this frame")
            return
        image = crop_face(image, landmarks)
    image = cv2.resize(image, shape)
    tensor = im2ten_tf(image)
    batch = torch.unsqueeze(tensor, 0)
    return batch

def image2features(image, detector):
    """
    Convert an image into facial keypoints features coming from a mesh face detector. 
    Mesh feature is returned as a Pytorch batch of 1 tensor.

    Args:
        image (np.ndarray): BGR frame from cv2 video
        detector (method): detector used to create facial mesh

    Returns:
        torch.Tensor: float tensor of shape (1, mesh_size)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = get_landmarks(image, detector)
    if landmarks is None: 
        print("No face detected in this frame")
        return
    features = list()
    rel_landmarks = detector.absolute_to_relative_landmarks(landmarks, shape=image.shape)
    for i in range(len(rel_landmarks.multi_face_landmarks[0].landmark)):
        point = [rel_landmarks.multi_face_landmarks[0].landmark[i].x,
                 rel_landmarks.multi_face_landmarks[0].landmark[i].y,
                 rel_landmarks.multi_face_landmarks[0].landmark[i].z
                 ] 
        features.append(point[0])
        features.append(point[1])
        features.append(point[2])
    features = torch.tensor(np.array(features), dtype=torch.float32)
    batch = torch.unsqueeze(features, 0)
    return batch

def loadNN(model_path, model_class=None, backbone=None, pretrained=True):
    """
    Load a neural network model, load trained weights, and return it for prediction.

    Args:
        model_path (str): Path to the trained weights to load into the model
        model_class (python class): Class type of the model to return
        backbone (str): Name of the backbone used to create and train the model.
        pretrained (_type_, optional): _description_. Defaults to True.

    Returns:
        `model_class`: the trained model
    """
    if model_class is None: 
        return None
    model = model_class(outputs=2,
                        backbone=backbone,
                        pretrained=pretrained,
                        model_name="Temp")
    model.loadModel(model_path)
    print(f"Loaded weights {model_path} into {model_class} with backbone {backbone}")
    model.train(False)
    return model

def loadML(model_path, model_name, parameters):
    """
    Load a machine learning model, load trained weights, and return it for prediction.
    Args:
        model_path (str): Path to the trained weights to load into the model
        model_name (str): Name of the model to return
        parameters (dict): Dictionary containing the parameters of the model
    
    Returns:
        Regressor: a model initialized with the given weights.
    """
    model = Regressor(model=model_name, **parameters)
    model.loadModel(model_path)
    print(f"Loaded weights {model_path} into {model_name} with parameters {parameters}")
    return model


if __name__ == "__main__":
    """
    test == "DRAW": draw the detected position on the video
    test == "MOVE": move the mouse to the detected position

    use_nn == True: use a neural network model
    use_nn == False: use a machine learning model
    """
    test = "MOVE"
    use_nn = False

    """
    DEFINE YOUR MODEL BELOW
    Comment the part that is not relevant
    """

    
    if use_nn:
        """
        Load Neural Network Model
        -------------------------
        """
        model_path = r"models/Custom_crop_face8.json"

        model = loadNN(model_path, 
                    model_class=FaceTrackerNN, 
                    backbone="CustomModel", 
                    pretrained=True)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        crop = "face"
        shape = (224, 224)
        detector = None
        if crop == "face":
            shape = (224, 224)
            detector = get_detector()
        elif crop == "eyes":
            shape = (64, 128)
    else:

        """
        Load Machine Learning Model
        -------------------------
        """
        model_path = r"models/debug1"
        model_name = "forest"
        parameters = {'forest_param': ["squared_error",50,100]}
        model = loadML(model_path, model_name, parameters)
        detector = get_detector()
    
    cap = cv2.VideoCapture(0)
    frame_width, frame_height = pyautogui.size()

    if not cap.isOpened():
        print("Error while reading camera feed.")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame!")
            break

        if use_nn:
            batch = image2batch(frame, shape, detector=detector)
            if batch is None:
                break
            if batch is not None:
                batch = batch.to(model.device)
                prediction = model.forward(batch).cpu()
                prediction = torch.squeeze(prediction).detach().numpy()
            else:
                prediction = None
        else:  # Machine Learning model
            features = image2features(frame, detector)
            prediction = model.forward(features)[0]

        if test == "DRAW":
            if prediction is not None:
                draw_point_on_frame(frame, prediction)
            else:
                print("No prediction...")

            cv2.imshow("Prediction (green)", frame)
            if (cv2.waitKey(10) & 0xFF) == ord('q'):
                break
        elif test == "MOVE":
            if prediction is None:
                continue
            X = prediction[0]*frame_width
            Y = prediction[1]*frame_height
            pyautogui.moveTo(X, Y)

        
    cap.release()
    cv2.destroyAllWindows()
