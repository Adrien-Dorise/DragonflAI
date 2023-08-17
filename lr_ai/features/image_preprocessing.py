"""
 Image preprocessing toolbox for data coming from the faceMask app
 Author: Julia Cohen - Adrien Dorise ({jcohen, adorise}@lrtechnologies.fr) - LR Technologies
 Created: March 2023
 Last updated: Adrien Dorise - August 2023
"""

import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from imgaug import augmenters as iaa

from lr_ai.features.tracker_toolbox import crop_face, crop_eyes, get_detector, get_landmarks
from lr_ai.features.preprocessing import getTrackerSet, load_metadata


# ----------------
# Definition of transforms for training and test

def get_tf(name="TF1", shape=224, padd=False, padd_shape=None):
    if padd:
        pshape = padd_shape if padd_shape is not None else shape
        if len(pshape) != 2:
            pshape = (pshape, pshape)
        padd_op = iaa.PadToFixedSize(width=pshape[1], height=pshape[0])
    else:
        padd_op = iaa.Noop()
    if name == "TF1":
        # Training tf
        return iaa.Sequential([
                iaa.Sometimes(p=0.3, then_list=[
                    iaa.Add((-40, 40), per_channel=0.2)
                ]),
                iaa.Sometimes(p=0.3, then_list=[
                    iaa.Grayscale()
                ]),
                iaa.Sometimes(p=0.2, then_list=[
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.015*255))
                ]),
                iaa.Sometimes(p=0.3, then_list=[
                    iaa.ChangeColorTemperature((5000, 7000))
                ]),
                iaa.Sometimes(p=0.5, then_list=[
                    iaa.Multiply((0.5, 1.2), per_channel=0.2),
                    iaa.MultiplySaturation((0.9, 1.1))
                ]),
                iaa.Resize(shape),
                padd_op
                ],
            random_order=False)
    elif name == "TF2":
        # Test tf
        return iaa.Sequential([
                padd_op,
                iaa.Resize(shape)
                ], 
                random_order=False)
    elif name == "TF3":
        # Single op tf (to test an augmentation)
        return iaa.Sequential([
                iaa.ChangeColorTemperature((5000, 7000)),
                iaa.Resize(shape),
                padd_op
                ],
                random_order=False)


class ImageDataset(Dataset):
    '''
    Class used to store the dataset handled by pytorch, for Image inputs
    '''
    def __init__(self, data_folder, train=True, crop=None, debug=False):
        '''
        ImageDataset class constructor.
        Parameters
        ----------
        data_folder: 
            folder with videos to use as source.
        train: bool
           whether the dataset corresponds to a training set or a test set. 
           Default: True. 
        crop: string
            Crop a portion of the images. Can be 'eyes' or 'face'. Default is None.
        debug: bool
            The ToTensor op is not applied, in order to visualize the batch with 
            opencv more easily. Default is False.
        Returns
        ----------
        None
        '''
        self.targets = []
            
        self.caps = []
        self.caps_lengths = []
        self.cumul_lengths = []
            
        self.targets = list()
        
        self.train = train
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.shape = (224, 224)
        padd = False
        padd_shape = None
        # TODO setup padding for cropped eyes
        if crop == 'eyes':
            self.crop_fct = crop_eyes
            self.detector = get_detector()
            #shape = (112, 224)
            #padd = True
            #padd_shape = (72, 2*72)
        elif crop == 'face':
            self.crop_fct = crop_face
            self.detector = get_detector()
            padd = False
        elif crop is None:
            self.crop_fct = None
            self.detector = None
        else:
            print(f"Got {crop} as 'crop' argument, should be 'face' or 'eyes' or None.")
            exit(1)
            
        if self.train:
            self.tf = get_tf("TF1", shape=self.shape, padd=padd, padd_shape=padd_shape)
        else:
            self.tf = get_tf("TF2", shape=self.shape, padd=padd, padd_shape=padd_shape)
        self.debug = debug

    def __len__(self):
        return  sum(self.caps_lengths)
    
    def __getitem__(self, idx):
        # Find the cap object holding frame of correct index
        frame = self.caps[idx]
        assert frame is not None, \
            f"Frame {idx} not loaded ({self.img_paths[idx]})."

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            if self.crop_fct is not None:
                landmarks = get_landmarks(frame, self.detector)
                while landmarks is None:  # no face detected
                    idx += 1
                    frame = self.caps[idx]
                    assert frame is not None, \
                        f"Frame {idx} not loaded ({self.img_paths[idx]})."
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks(frame, self.detector)

                frame = self.crop_fct(frame, landmarks)
        except AttributeError as e:
            print(e)
            print("No face detected, bug should be handled but seems like it is not.")
            print(f"Error comes from image {self.img_paths[idx]}.")
            exit()
        
        frame = self.tf(image=frame)
        if not self.debug:
            frame = self.to_tensor(frame)
        
        return frame, self.targets[idx]

    def __del__(self):
        for cap in self.caps:
            del cap

class ImageClassificationDataset(ImageDataset):
    '''
    Class used to store the dataset handled by pytorch, for Image inputs
    '''
    def __init__(self, data_folder, train=True, crop=None, debug=False):
        '''
        ImageDataset class constructor.
        Parameters
        ----------
        data_folder: 
            folder with videos to use as source.
        train: bool
           whether the dataset corresponds to a training set or a test set. 
           Default: True. 
        crop: string
            Crop a portion of the images. Can be 'eyes' or 'face'. Default is None.
        debug: bool
            The ToTensor op is not applied, in order to visualize the batch with 
            opencv more easily. Default is False.
        Returns
        ----------
        None
        '''
        self.img_paths = [os.path.join(root, file) for root, dirs, files in os.walk(data_folder)
             for file in files if file.endswith(".jpg")]
        if len(self.img_paths) == 0:
            print("No img file found")
            exit(0)

        #get name of the subfolders
        classes = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]

        # create a dict that for each class associate a number
        dict_classes = {classes[i]: i for i in range(len(classes))}

        targets = []

        #for each image, add the corresponding target to the targets list
        for image_path in self.img_paths:
            targets.append(dict_classes[os.path.basename(os.path.dirname(image_path))])
            
        self.caps = [cv2.imread(path) for path in self.img_paths]
        self.caps_lengths = [1] * len(self.caps)
        self.cumul_lengths = np.cumsum(self.caps_lengths)
            
        self.targets = list()
        for i, target in enumerate(targets):
            if not torch.is_tensor(target):
                self.targets.append(torch.tensor(target))
            else:
                self.targets.append(target)
        
        self.train = train
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.shape = (224, 224)
        padd = False
        padd_shape = None
        # TODO setup padding for cropped eyes
        if crop == 'eyes':
            self.crop_fct = crop_eyes
            self.detector = get_detector()
            #shape = (112, 224)
            #padd = True
            #padd_shape = (72, 2*72)
        elif crop == 'face':
            self.crop_fct = crop_face
            self.detector = get_detector()
            padd = False
        elif crop is None:
            self.crop_fct = None
            self.detector = None
        else:
            print(f"Got {crop} as 'crop' argument, should be 'face' or 'eyes' or None.")
            exit(1)
            
        if self.train:
            self.tf = get_tf("TF1", shape=self.shape, padd=padd, padd_shape=padd_shape)
        else:
            self.tf = get_tf("TF2", shape=self.shape, padd=padd, padd_shape=padd_shape)
        self.debug = debug
    
    def __getitem__(self, idx):
        # Find the cap object holding frame of correct index
        frame = self.caps[idx]
        assert frame is not None, \
            f"Frame {idx} not loaded ({self.img_paths[idx]})."

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            if self.crop_fct is not None:
                landmarks = get_landmarks(frame, self.detector)
                while landmarks is None:  # no face detected
                    idx += 1
                    frame = self.caps[idx]
                    assert frame is not None, \
                        f"Frame {idx} not loaded ({self.img_paths[idx]})."
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks(frame, self.detector)

                frame = self.crop_fct(frame, landmarks)

        except AttributeError as e:
            print(e)
            print("No face detected, bug should be handled but seems like it is not.")
            print(f"Error comes from image {self.img_paths[idx]}.")
            exit()
        
        frame = self.tf(image=frame)
        if not self.debug:
            frame = self.to_tensor(frame)

        
        return frame, self.targets[idx]

    def __del__(self):
        for cap in self.caps:
            del cap

class ImageRegressionDataset(ImageDataset):
    '''
    Class used to store the dataset handled by pytorch, for Image inputs
    '''
    def __init__(self, data_folder, train=True, crop=None, debug=False):
        '''
        ImageDataset class constructor.
        Parameters
        ----------
        data_folder: 
            folder with videos to use as source.
        train: bool
           whether the dataset corresponds to a training set or a test set. 
           Default: True. 
        crop: string
            Crop a portion of the images. Can be 'eyes' or 'face'. Default is None.
        debug: bool
            The ToTensor op is not applied, in order to visualize the batch with 
            opencv more easily. Default is False.
        Returns
        ----------
        None
        '''
        self.video_paths = videos_in_folder(data_folder)
        if len(self.video_paths) == 0:
            print("No avi file found")
            exit(0)
        
        targets_csv_list = [v.replace('.avi', '.csv').replace('user', 'mouse') 
                            for v in self.video_paths]
        metadatas_list = [t.replace("mouse", "metadata") for t in targets_csv_list]
        targets = [getTrackerSet(None, target_csv)[1] for target_csv in targets_csv_list]  # targets in absolute values
        self.metadatas = [load_metadata(meta_csv) for meta_csv in metadatas_list]  # [(w, h),...]
        
        self.caps = [video2cap(path) for path in self.video_paths]
        self.caps_lengths = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in self.caps]
        self.cumul_lengths = np.cumsum(self.caps_lengths)
        
        self.targets = list()
        for i, target in enumerate(targets):
            target[:, 0] /= self.metadatas[i][0]
            target[:, 1] /= self.metadatas[i][1]
            if not torch.is_tensor(target):
                self.targets.append(torch.from_numpy(target))
            else:
                self.targets.append(target)
        
        self.train = train
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),  # convert from [0, 255] to [0.0, 0.1]
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.shape = (224, 224)
        padd = False
        padd_shape = None
        # TODO setup padding for cropped eyes
        if crop == 'eyes':
            self.crop_fct = crop_eyes
            self.detector = get_detector()
            #shape = (112, 224)
            #padd = True
            #padd_shape = (72, 2*72)
        elif crop == 'face':
            self.crop_fct = crop_face
            self.detector = get_detector()
            padd = False
        elif crop is None:
            self.crop_fct = None
            self.detector = None
        else:
            print(f"Got {crop} as 'crop' argument, should be 'face' or 'eyes' or None.")
            exit(1)
        
        if self.train:
            self.tf = get_tf("TF1", shape=self.shape, padd=padd, padd_shape=padd_shape)
        else:
            self.tf = get_tf("TF2", shape=self.shape, padd=padd, padd_shape=padd_shape)
        self.debug = debug
    
    def __getitem__(self, idx):
        # Find the cap object holding frame of correct index
        cap_idx = len(np.where(self.cumul_lengths<=idx)[0])
        if cap_idx > 0:
            frame_idx = idx - self.cumul_lengths[cap_idx-1]
        else:
            frame_idx = idx
        frame = cap2frame(cap=self.caps[cap_idx], index=frame_idx)
        assert frame is not None, \
            f"Frame {frame_idx} from video {cap_idx} not loaded ({self.video_paths[cap_idx]})."

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            if self.crop_fct is not None:
                landmarks = get_landmarks(frame, self.detector)
                while landmarks is None:  # no face detected
                    frame_idx += 1
                    assert frame_idx<self.caps_lengths[cap_idx], \
                        f"Face not detected and frame_idx has max_value ({frame_idx}) for video {self.video_paths[cap_idx]}."
                    frame = cap2frame(cap=self.caps[cap_idx], index=frame_idx)
                    assert frame is not None, \
                        f"Frame {frame_idx} from video {cap_idx} not loaded ({self.video_paths[cap_idx]})."
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks = get_landmarks(frame, self.detector)

                frame = self.crop_fct(frame, landmarks)
        except AttributeError as e:
            print(e)
            print("No face detected, bug should be handled but seems like it is not.")
            print(f"Error comes from video {self.video_paths[cap_idx]} around frame {frame_idx}.")
            exit()
        
        frame = self.tf(image=frame)
        if not self.debug:
            frame = self.to_tensor(frame)
        
        return frame, self.targets[cap_idx][frame_idx]

    def __del__(self):
        for cap in self.caps:
            cap.release()


class VideoDataset(ImageDataset):
    '''
    Class used to store the dataset handled by pytorch, for Video inputs
    '''
    def __init__(self, data_folder, seq_length, train=True, crop=None, debug=False):
        '''
        ImageDataset class constructor.
        Parameters
        ----------
        data_folder: 
            folder with videos to use as source.
        seq_length: 
            Number of previous frame to use as temporal input
        train: bool
           whether the dataset corresponds to a training set or a test set. 
           Default: True. 
        crop: string
            Crop a portion of the images. Can be 'eyes' or 'face'. Default is None.
        debug: bool
            The ToTensor op is not applied, in order to visualize the batch with 
            opencv more easily. Default is False.
        Returns
        ----------
        None
        '''
        super().__init__(data_folder=data_folder, train=train, crop=crop, debug=debug)
        self.seq_length = seq_length
      
        
    def __getitem__(self, idx):
        frames_tensor = torch.Tensor(self.seq_length, 3, self.shape[0], self.shape[1])
        frames = []

        if idx >= self.seq_length - 1:
            idx_start = idx - self.seq_length + 1
            for i in range(idx_start,idx+1):
                frame,_ = super().__getitem__(i)
                frames.append(frame.reshape(1,frame.shape[0], frame.shape[1], frame.shape[2]))
        else:
            for i in range(0,self.seq_length):
                frame,_ = super().__getitem__(i)
                frames.append(frame.reshape(1,frame.shape[0], frame.shape[1], frame.shape[2]))
        
        frames_tensor = torch.cat(frames)
        _,target = super().__getitem__(idx)
        
        
        #print(np.shape(frames_tensor))
        #print(f"1: {frames_tensor[0]}")
        #print(f"2: {frames_tensor[1]}")

        #frame = super().__getitem__(0)
        #print(f"1: {frame}")
        #frame = super().__getitem__(1)
        #print(f"2: {frame}")

        
        return frames_tensor, target
        
        return frame, self.targets[cap_idx][frame_idx]


def videos_in_folder(folder):
    assert os.path.isdir(folder), f"Path {folder} is not a directory"
    video_paths = [os.path.join(folder, file) for file in os.listdir(folder) 
                   if file.endswith(".avi")]
    return video_paths

def images_in_folder(folder):
    assert os.path.isdir(folder), f"Path {folder} is not a directory"
    img_paths = [os.path.join(folder, file) for file in os.listdir(folder) 
                   if file.endswith(".jpg")]
    return img_paths

def cap2frame(cap, index):
    '''
    Select and return a frame from a video stream by index.
    Parameters
    ----------
    cap: cv2.VideoCapture object, already initialized on the desired video source
    index: index of the frame to extract. 
            /!\ It must have been verified BEFORE that the index is valid, for example
            /!\ using totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    Returns
    ----------
    frame: numpy array, frame in BGR colorspace, not flipped
        or None if the frame could not be read.
    '''
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        return None

def video2cap(video_file):
    '''
    Opens a video file and returns the corresponding video capture object.
    Parameters
    ----------
    videoFile: string
         Path to the video file.
    Returns
    ----------
    cap: cv2.VideoCapture object
        or None if the video could not be opened.
    '''
    if not os.path.isfile(video_file):
        raise Warning(f"Can't load data: video file {video_file} does not exist")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open video file")
    return cap

def image_collate_fn(data):
    """
    Custom collate function that returns the images and targets 
    from a batch in the form of 2 tuples, instead of a single sequence.

    Parameters
    ----------
    data: list
        Sequence of (image, target) produced by the dataloader.
    Returns
    ----------
    imgs: list
        List of images (size as the batch size of the dataloader).
    targets: list
        List of 
    """
    imgs, targets = zip(*data)  # tuples
    return torch.stack(imgs, 0), torch.stack(targets, 0)

if __name__ == "__main__":
    folder = './data/split1/train'
    dataset = ImageRegressionDataset(folder, train=False, crop=None, debug=False)

    print(f"Dataset of size {len(dataset)}")

    for i in np.random.randint(0, len(dataset), 10):
        img, target = dataset[i]
        np_img = img.permute(1, 2, 0).numpy()
        h, w, _ = np_img.shape
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        x = int((1.0-target[0].numpy())*w)  # Revert left-right
        y = int(target[1].numpy()*h)
        cv2.circle(bgr, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
        cv2.imshow("frame", bgr)
        if (cv2.waitKey(0) & 0xFF) == ord('q'):
                break
    cv2.destroyAllWindows()

    # NUM WORKERS HAS TO BE 1, otherwise openCV does not manage to read the frames
    dloader = DataLoader(dataset, batch_size=4, collate_fn=image_collate_fn, num_workers=1)
    for i, data in enumerate(dloader):
        print(data[0].shape)
        if i >= 5:
            break

def img_loader(folder_path, 
               isTrain, 
               batch_size=16, 
               crop=None, 
               shuffle=True, 
               num_workers=0, 
               temporal=False,
               classification=False, 
               sequence_length=2,):
    """Create Pytorch DataLoader object from array dataset
    
    Args:
        features (array of size (inputs, features)): set containing the features values
        target (set of size (inputs, targets)): Set containing the targets values
        batch_size (int, optional): batch_size used for DataLoader. Defaults to 16.
        no_batch (bool, optional): False if user wants trainset/testset divided in mini batches (mostly used to train neural networks). True if user want a unqiue batch for all set (mostly used for machine learning). Default is False
        shuffle (bool, optional): True to shuffle the dataset. Defaults to True
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default to 0
        temporal (bool, optional): True to create a Loader incorporating previous observations in each samples. Defaults to False
        sequence_length (int, optional): Only used when temporal set to True. Number of previous observation to take in the model. Default to 1
        
    Returns:
        dataset: Pytorch DataLoader object
    """
    if(temporal):
        shuffle=False
        dataset = VideoDataset(folder_path,  
                                seq_length=sequence_length, 
                                train=isTrain, 
                                crop=crop, 
                                debug=False)
    elif(classification):
        dataset = ImageClassificationDataset(folder_path, 
                                train=isTrain, 
                                crop=crop, 
                                debug=False)
    else:
        dataset = ImageRegressionDataset(folder_path, 
                                train=isTrain, 
                                crop=crop, 
                                debug=False)
        
    loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            collate_fn=image_collate_fn,
                            shuffle=shuffle)
    return loader
