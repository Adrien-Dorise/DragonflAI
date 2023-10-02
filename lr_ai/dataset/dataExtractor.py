#!/usr/bin/env python3
# coding: utf-8

# license : open sources
# Author Villain Edouard

import numpy as np 
import h5py 
import os 


def get_data_from_dataloader(loader):
    X = []
    Y = []
    nb_samples = 0
    for batch_ndx, sample in enumerate(loader):
        for i in range(len(sample[0])):
            X.append(np.array(sample[0][i]))
            Y.append(np.array(sample[1][i]))

            print("adding sample {} ".format(nb_samples), end='\r')
            nb_samples += 1
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def compute_13xy(path):
    import faceandmouse.features.preprocessing as pr
    from sklearn.preprocessing import MinMaxScaler
    train_set, _ = pr.loader(path, 
                             no_batch=True, 
                             batch_size=32, 
                             scaler=MinMaxScaler(), 
                             coords='xy', 
                             num_workers=0, 
                             tracker_version=1, 
                             temporal=False, 
                             sequence_length=0)
    
    X, Y = get_data_from_dataloader(train_set)
    os.makedirs('{}/13xy'.format(path))
    with h5py.File('{}/13xy/x.h5'.format(path), 'w') as f:
        f.create_dataset('x', data=X)
        
    with h5py.File('{}/13xy/y.h5'.format(path), 'w') as f:
        f.create_dataset('y', data=Y)
        

def compute_13xyz(path):
    import faceandmouse.features.preprocessing as pr
    from sklearn.preprocessing import MinMaxScaler
    train_set, _ = pr.loader(path, 
                             no_batch=True, 
                             batch_size=32, 
                             scaler=MinMaxScaler(), 
                             coords='xyz', 
                             num_workers=0, 
                             tracker_version=1, 
                             temporal=False, 
                             sequence_length=0)
    
    X, Y = get_data_from_dataloader(train_set)
    os.makedirs('{}/13xyz'.format(path))
    
    with h5py.File('{}/13xyz/x.h5'.format(path), 'w') as f:
        f.create_dataset('x', data=X)
        
    with h5py.File('{}/13xyz/y.h5'.format(path), 'w') as f:
        f.create_dataset('y', data=Y)