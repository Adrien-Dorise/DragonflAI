"""
 Preprocessing toolbox for data coming from the faceMask app
 Author: Julia Cohen - Adrien Dorise ({jcohen, adorise}@lrtechnologies.fr) - LR Technologies
 Created: Feb 2023
 Last updated: Adrien Dorise - July 2023
"""

import os

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
import sklearn.utils as utils
import torch
from torch.utils.data import Dataset, DataLoader

from camerasensor import MeshFaceDetector
from lr_ai.features.tracker_toolbox import select_points


class Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch. 
    '''
    def __init__(self, feat, target):
        if not torch.is_tensor(feat):
            self.feat = torch.from_numpy(feat)
        else:
            self.feat = feat
        
        if not torch.is_tensor(target):
            self.target = torch.from_numpy(target)
        else:
            self.target = target

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        return self.feat[idx], self.target[idx]

class Sequential_Dataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch.
    It is designed to handle sequential datasets (timeSeries, video...) 
    '''
    def __init__(self, feat, target, seq_length):
        if not torch.is_tensor(feat):
            self.feat = torch.from_numpy(feat)
        else:
            self.feat = feat
        
        if not torch.is_tensor(target):
            self.target = torch.from_numpy(target)
        else:
            self.target = target
        
        self.seq_length = seq_length

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, idx):
        #For sequential data, we the loader so that each sample adds up with previous ones
        #When we are at the beginning of the dataset, we use only first iteration
        if idx >= self.seq_length - 1:
            idx_start = idx - self.seq_length + 1
            feat = self.feat[idx_start:(idx + 1), :]
        else:
            padding = self.feat[0].repeat(self.seq_length - idx - 1, 1)
            feat = self.feat[0:(idx + 1), :]
            feat = torch.cat((padding, feat), 0)
        return feat, self.target[idx]


def write_csv(data, folder_path, file_name='newFile'):
    '''
    Save numpy data as a csv file. The specified folder is created if does not already exists.
    Parameters
    ----------
    folder_path : string
        Path to the folder to save the csv file.
    file_name : string, optional
        Name of the file to save. If there is no extension, a .csv will be added.
        If a file with same name already exists, a numerical index is added.
        Default: newFile
    Returns
    -------
    None.
    '''
    if folder_path == '':
        raise Warning("Can't save data: folder path not specified")
        
    extension = ".csv"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    current_ext = os.path.splitext(file_name)[-1]
    if current_ext != "":
        file_name = file_name.replace(current_ext, "")
    
    increment = 1
    file_path = os.path.join(folder_path, f"{file_name}{extension}") 
    while os.path.isfile(file_path):
        file_path = os.path.join(folder_path, f"{file_name}{increment}{extension}")
        increment += 1
    if(type(data) != np.ndarray):
        data = np.asarray(data)
    np.savetxt(file_path, data, delimiter = ',')
    print('file saved in: ' + file_path)


def load_csv(file_path, selectCol=None):
    '''
    Load a CSV file. 
    Parameters
    ----------
    file: string
        Path to the CSV file.
    selectCol: list or None
        Indices of the columns to extract from the csv
    Returns
    -------
    Numpy array containing the loaded data
    '''
    if not os.path.isfile(file_path):
        raise Warning(f"Can't load data: file {file_path} does not exist")
    if file_path[-4:] != ".csv":
        raise Warning("Can't load data: given file is not CSV file")
    
    try:
        return np.loadtxt(file_path, delimiter=',', skiprows=1, dtype=None, usecols=selectCol)
    except ValueError:
        # Last row of mouse.csv files has a string in the last selected column --> ValueError
        return np.genfromtxt(file_path, delimiter=',', skip_header=1, skip_footer=1, usecols=selectCol)


def video2tracker(video_file, save_folder):
    '''
    Process a video to extract face points position. The positions are stored in a CSV file inside the given saveFolder
    Parameters
    ----------
    videoFile: string
         Path to the video file.
     saveFolder
         Path to the folder to store the process data. The name of the CSV file is similar to the video file.
    Returns
    -------
    Numpy array containing the loaded data
    '''
    if not os.path.isfile(video_file):
        raise Warning(f"Can't load data: video file {video_file} does not exist")
    
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    save_path = os.path.join(save_folder, os.path.basename(video_file).replace(os.path.splitext(video_file)[-1], ".csv"))
    detector = MeshFaceDetector(refine_landmarks=True)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open video file")
    print("Video file open!")
    try:
        ret, frame = cap.read()
        fill_cols = True
        all_data = list()
        columns = list()
        while ret:
            data = list()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame, roi=None)
            landmarks = detector.postprocess_detections(landmarks)
            if landmarks is None:
                data.append(-1)
                data.append(-1)
                data.append(-1)

                all_data.append(data)
                ret, frame = cap.read()
                continue

            rel_landmarks = detector.absolute_to_relative_landmarks(landmarks, shape=rgb_frame.shape)
            
            for i in range(len(rel_landmarks.multi_face_landmarks[0].landmark)):
                # Get all points from one frame
                point = [
                rel_landmarks.multi_face_landmarks[0].landmark[i].x,
                    rel_landmarks.multi_face_landmarks[0].landmark[i].y,
                    rel_landmarks.multi_face_landmarks[0].landmark[i].z
                    ] 
                data.append(point[0])
                data.append(point[1])
                data.append(point[2])
                if fill_cols:
                    columns.append(f'pt{i}_x')
                    columns.append(f'pt{i}_y')
                    columns.append(f'pt{i}_z')
            fill_cols = False
            all_data.append(data)
            
            ret, frame = cap.read()

        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(save_path)
    finally:
        cap.release()

def getTrackerSet(trackerPath, targetPath):
    """Load tracker + mouse CSV files into Python variables

    Args:
        trackerPath (string): Path to the tracker CSV
        targetPath (string): Path to the mouse CSV

    Returns:
        array of size (inputs, features): array containing the trackers results
        array of size (inputs, targets): array containing mouse positions        
    """
    featuresTracker= None
    if trackerPath is not None:
        featuresTracker = load_csv(trackerPath)
        featuresTracker = featuresTracker[:,1:].astype(float)
    target = load_csv(targetPath, selectCol=[2,3,7])  # also select Camera sync column
    #print("Tracker shape=", featuresTracker.shape)
    #print("Target shape=", target.shape)  # Shapes mismatch because target contains all mouse events
    
    # Selection of mouse events corresponding to frames
    target = selectEventsWithFrames(target)
    
    return featuresTracker, target[:, :2]

def selectEventsWithFrames(target):
    """
    Removes data entries not corresponding to a frame +
    Replaces empty values (still mouse) with previous position instead of the default (0, 0) +
    Filters duplicated entries: due to synchro issues in UserRecorderApp, some mouse events 
    are present twice with exact same frame timestamp (instead of one with a 0 and one with a timestamp).

    Args:
        target (np.ndarray): data array as read by load_csv, with 3 columns (X, Y, Camera Sync)

    Returns:
        np.ndarray: data array with 3 columns (X, Y, Camera Sync), but only relevant rows.
    """
    keep_indices = target[:, 2] != 0
    target = target[keep_indices, :]
    still_indices = (target[:,:2] == 0)
    still_indices = np.logical_and(still_indices[:,0], still_indices[:,1])
    still_indices= np.where(still_indices == True)[0]
    for k in still_indices:
        target[k,:2] = target[k-1,:2]
    _, ind = np.unique(target, axis=0, return_index=True)
    ind.sort()
    return target[ind, :]

def dataSplit(features, target, split=0.8, batch_size = 16, no_batch = False, shuffle = True):
    """Create train and test set from a unique data set

    A simple split is performed. Both train and test set are then inserted into Pytorch DataLoader
    
    Args:
        features (array of size (inputs, features)): set containing the features values
        target (set of size (inputs, targets)): Set containing the targets values
        split (float, optional): Proportion of inputs taken for the trainset. Defaults to 0.8.
        batch_size (int, optional): batch_size used for DataLoader. Defaults to 16.
        no_batch (bool, optional): False if user wants trainset/testset divided in mini batches (mostly used to train neural networks). True if user want a unqiue batch for all set (mostly used for machine learning). Default is False
        shuffle (bool, optional): True to shuffle the dataset. Defaults to True
    Returns:
        DataLoader: Training set
        DataLoader: Testing set.
    """

    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=split, shuffle=shuffle)
        
    train_set = loader(X_train,y_train, batch_size, no_batch, shuffle=False)
    test_set = loader(X_test, y_test, batch_size, no_batch, shuffle=False)   
    
    return train_set, test_set

def loader(folder_path, 
           batch_size=16, 
           no_batch=False, 
           shuffle=True, 
           scaler=None, 
           coords="xyz", 
           tracker_version=1, 
           num_workers=1, 
           temporal=False, 
           sequence_length=1):
    """Create Pytorch DataLoader object from array dataset
    
    Args:
        folder_path (string): Path to the folder containing the videos + target needed (ex: dataset/data1)
        batch_size (int, optional): batch_size used for DataLoader. Defaults to 16.
        no_batch (bool, optional): False if user wants trainset/testset divided in mini batches (mostly used to train neural networks). True if user want a unqiue batch for all set (mostly used for machine learning). Default is False
        shuffle (bool, optional): True to shuffle the dataset. Defaults to True
        scaler (sklearn.preprocessing): Type of scaler used for the data (ex: MinMaxScaler()). Put None if you do not want to scale the data. If the scaler is not fitted yet, it will be on the data given with folder_path. Default to None.
        coords: str, optional
        specify if we keep the 3D coordinates ('xyz') or only 2D 
        coordinates ('xy'). Defaults to "xy".
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
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default to 0
        temporal (bool, optional): True to create a Loader incorporating previous observations in each samples. Defaults to False
        sequence_length (int, optional): Only used when temporal set to True. Number of previous observation to take in the model. Default to 1
    Returns:
        dataset: Pytorch DataLoader object
        scaler (sklearn.preprocessing object): Scaler that was used to fit the features. Use it to fit other dataset instances
    """

    videos = os.listdir(folder_path)
    videos = [folder_path+'/'+vid for vid in videos if vid.endswith(".avi")]
    features, targets = multi_file_set(videos)
    features = select_points(features, coords=coords, version=tracker_version)
    print(f"Feature selection: shape={features.shape} - Targets shape={targets.shape}")
    if(scaler is not None):
        try:
            features = scaler.transform(features)
        except NotFittedError as e:
            features = scaler.fit_transform(features)
        

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    
    if(shuffle and not temporal):
        features, targets = utils.shuffle(features, targets)
    
    if(no_batch):
        batch_size = np.shape(features)[0]
    
    if(temporal):
        dataset = Sequential_Dataset(features,targets,sequence_length) 
    else:
        dataset = Dataset(features,targets)
   
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return loader,scaler

def load_metadata(metadata_path):
    """
    Read the screen size in the metadata csv file

    Args:
        metadata_path (str): path to the metadata file

    Returns:
        width: int
        height: int
    """
    data = load_csv(metadata_path, selectCol=[2])
    width = data[0]
    height = data[1]
    return width, height

def multi_file_set(videos_list):
    """
    Concatenates multiple sources of data into a single data array

    Args:
        videos_list (list of strings): list of paths to the videos to use

    Returns:
        all_features (np.array): all input features as a single numpy array
        all_targets (np.array): all targets as a single numpy array
    """
    targets_csv_list = [v.replace('.avi', '.csv').replace('user', 'mouse') for v in videos_list]
    metadatas_list = [t.replace("mouse", "metadata") for t in targets_csv_list]
    feat_tracker_list = [v.replace(".avi", ".csv") for v in videos_list]
    features_list = list()
    targets_list = list()
    for v, t, m, f in zip(videos_list, targets_csv_list, metadatas_list, feat_tracker_list):
        print(f"\nNew video: {os.path.basename(v)}")
        if not os.path.isfile(f):
            output_path = os.path.dirname(v)
            # Results of tracker saved in same folder as video
            video2tracker(v, output_path)
        feature, target = getTrackerSet(f, t)
        # print(f"Feature shape={feature.shape} - Target shape={target.shape}")
        w, h = load_metadata(m)
        target[:, 0] = target[:, 0]/w
        target[:, 1] = target[:, 1]/h
        features_list.append(feature)
        targets_list.append(target)
    all_features = np.vstack(features_list)
    all_targets = np.vstack(targets_list)
    print(f"\nDone!\nFeatures shape={all_features.shape} - Targets shape={all_targets.shape}")
    return all_features, all_targets


if __name__ == "__main__":
    TEST = "multi_file"

    if TEST == "csv":
        a = np.array([1,2,3,4])

        folderPath = '../data/Debug'
        filePath = '/Debug.csv'

        write_csv(a,folderPath,filePath)
        b = load_csv(folderPath + filePath)
    elif TEST == "tracker":
        video2tracker(r'data/user 2023-03-21 142117.avi', 'output_folder')
        
    elif TEST == "load_data":
        a,b = getTrackerSet("output_folder/user 2023-03-21 142117.csv", "data/mouse 2023-03-21 142117.csv")
        print(np.shape(a))
        print(np.shape(b))
        c,d = dataSplit(a,b)
        print(c)
        print(d)
    
    elif TEST == "select":
        a,b = getTrackerSet("output_folder/user 2023-03-21 142117.csv", "data/mouse 2023-03-21 142117.csv")
        print(np.shape(a))
        print(np.shape(b))
        a_bis = select_points(a, coords='xy', version=1)
        print(np.shape(a_bis))
        c,d = dataSplit(a_bis,b)
        print(c)
        print(d)
    
    elif TEST == "multi_file":
        paths = [
            r"../../data/user 2023-03-23 103131.avi",
            r"../../data/user 2023-03-23 103443.avi"
        ]
        features, targets = multi_file_set(paths)
        print(features.shape)
        print(targets.shape)
        train_set, test_set = dataSplit(features, targets, no_batch=True, split=0.8)
        
    