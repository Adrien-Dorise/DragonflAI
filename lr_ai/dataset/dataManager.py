#!/usr/bin/env python3
# coding: utf-8

# license : open sources
# Author Villain Edouard

import h5py  
import os
import json 
import numpy as np 

class DatasetFolder(object):
    
    def __init__(self, root):
        super(DatasetFolder, self).__init__()

        assert(os.path.isfile('{}/infos.json'.format(root))), \
            'Error, infos.json not found'
            
        self.root = root
        with open('{}/infos.json'.format(self.root), 'r') as mj:
            info = mj.read()
        self.info = json.loads(info)
        
        self.dataset_size = self.info['size']
        self.dataset = {}
        self.modalities = self.info['modalities']
        self.used_modalities = []
        
        for i in range(self.dataset_size):
            # create new dataset
            self.dataset[i] = DataFolder(i)
            
    def load_modality(self, idx, modality):
        '''
        load data in modalityBIDS file for each participant
        '''
        # assert modality exists
        assert(modality in self.modalities), \
            'Error, modality does not exists'

        # for each data in dataset
        for id, data in self.dataset.items():
            if id in idx:
                # load modality
                data.load_modality(modality)
        self.used_modalities.append(modality) 


    def load(self, idx, modalities=None):
        '''
        load data in modalityBIDS file for each participant
        '''
        # if no modalities provided : load all modalities
        if modalities is None:
            for modality in self.modalities:
                self.load_modality(idx, modality)
        else:
            # if one modality load it
            if isinstance(modalities, str):
                self.load_modality(idx, modalities)
            # if list of modalities, load them
            elif isinstance(modalities, list):
                for modality in modalities:
                    self.load_modality(idx, modality)
                    
    def set_modalities(self, modalities=None):
        if modalities is None:
            for i in range(self.dataset_size):
                self.dataset[i].set_modalities(self.modalities)
        else:
            # if one modality load it
            if isinstance(modalities, str):
                for i in range(self.dataset_size):
                    self.dataset[i].set_modalities(modalities)
            # if list of modalities, load them
            elif isinstance(modalities, list):
                for i in range(self.dataset_size):
                    self.dataset[i].set_modalities(modalities)
                    
                    
    def get(self, idx=None):
        if idx == None:
            id = [i for i in range(self.dataset_size)]
        else:
            id = idx
                
        # init empty list 
        x = []
        y = []
        # foreach modality 
        for j in range(len(self.modalities)):
            # if current modality is used 
            if self.modalities[j] in self.used_modalities:
                # init X and Y with first data 
                X = self.dataset[id[0]].data[self.modalities[j]].X
                Y = self.dataset[id[0]].data[self.modalities[j]].Y
                # concat rest of data 
                for i in id[1:]:
                    X = np.concatenate((X, self.dataset[i].data[self.modalities[j]].X))
                    Y = np.concatenate((Y, self.dataset[i].data[self.modalities[j]].Y))
                
                # append to final list 
                x.append(X)
                y.append(Y)
        return x, y 
            
    def get_idx(self, filter_name, filter_values):
        # init empty list 
        idx = []
        # foreach data 
        for i in range(self.dataset_size):
            # if user id of current data in user_ID list 
            if self.dataset[i].info[filter_name] in filter_values:
                # append it 
                idx.append(i)
        return idx 
    
    def add_folder(self, path, infos, funcs):
        # current folder to add 
        name = './data/{:04d}'.format(self.info['size'])
        # create new folder 
        os.makedirs(name)
        # increase size of dataset 
        self.info['size'] += 1 
        # Writing to info.json of dataset 
        with open('./data/infos.json', 'w') as outfile:
            json.dump(self.info, outfile)
        # copy files inside dataset 
        os.system('cp {}/* {}/'.format(path, name))
        # Writing to info.json of data 
        infos['path'] = name 
        with open('{}/infos.json'.format(name), 'w') as outfile:
            json.dump(infos, outfile)
            
        for func in funcs:
            func(name)
    
class DataFolder(object):
    """docstring for DatasetBIDS."""

    def __init__(self, ID, label=None, label_name=None):
        super(DataFolder, self).__init__()
        self.ID = ID
        self.label = label
        self.label_name = label_name
        self.root = 'data/{:04d}'.format(self.ID)
        # laod data json file
        with open('{}/infos.json'.format(self.root), 'r') as mj:
            info = mj.read()
        self.info = json.loads(info)


    def set_modalities(self, modalities):
        '''
        save data available modalities
        '''
        self.modalities = modalities
        self.data = {}
        self.modalities_loaded = {}
        self.check_data()
        self.init_modalities()


    def check_data(self):
        '''
        check modalities availability
        '''
        # check if data has infos.json
        assert(os.path.isfile('{}/infos.json'.format(self.root))), \
            'Error, Data {}/infos.json not found'.format(self.root)

        # check if all modalities are available
        for modality in self.modalities:
            # must have file : <modality>/x.h5 and <modality>/y.h5
            # create filename
            path = '{}/{}/'.format(self.root, modality)
            # assert file exists
            assert(os.path.isfile('{}x.h5'.format(path))), \
                'Error, {}x.h5 not found'.format(path)
            assert(os.path.isfile('{}y.h5'.format(path))), \
                'Error, {}y.h5 not found'.format(path)


    def init_modalities(self):
        '''
        init Modality for each modality
        '''
        for modality in self.modalities:
            path = '{}/{}/'.format(self.root, modality)
            self.data[modality] = Modality(modality, path)
            self.modalities_loaded[modality] = False



    def print_dataFolder_info(self):
        '''
        print data information
        '''
        for key, value in self.info.items():
            print('{} : {}'.format(key, value))


    def load_modality(self, modality):
        '''
        load data in modalityBIDS file
        '''
        self.data[modality].load()
        self.modalities_loaded[modality] = True


    def free_modality(self, modality):
        '''
        free data in modalityBIDS file
        '''
        self.data[modality].free_modality()
        self.modalities_loaded[modality] = False


class Modality(object):
    """docstring for DatasetBIDS."""

    def __init__(self, modality_name, path):
        super(Modality, self).__init__()

        self.modality_name = modality_name
        self.path = path
        self.X = None 
        self.Y = None 

    def load(self):
        '''
        load data of modality
        '''
        # load data
        with h5py.File('{}x.h5'.format(self.path), 'r') as f:
            self.X = f['x'][:]
        with h5py.File('{}y.h5'.format(self.path), 'r') as f:
            self.Y = f['y'][:]
        # save shape
        self.shape = self.X.shape

    def free(self):
        '''
        free memory, but keep path (load available after free)
        '''
        self.data = None
        self.shape = None
        
def myFunc():
    print('I do not compute anything')
    print('If you want to compute a derivated index')
    print('implement your code here')
           
if __name__ == '__main__':
    # create a object datasetFolder 
    # assert that inside 'data' folder everything is ok 
    # by using json file describing dataset and each data 
    # create empty object foreach folder inside 'data' 
    d = DatasetFolder('data')
    # here we want to use 2 derivated indexes from raw data (video in this case)
    # user provide a list of derivated indexes's names 
    # assert that everything is available 
    # every path is set 
    # create empty object foreach modalities foreach data 
    d.set_modalities(['13xy', '13xyz'])
    # select idx by using a filter name and some filter values 
    # filter_values is always a list of value 
    idx = d.get_idx(filter_name='user_ethnie', 
                            filter_values=['Freddie Mercury sosie', 'Compagnie Créole lover'])
    # load data for those idx and modalities specified 
    # d.dataset[idx].data[modality].X is loaded 
    d.load(idx, ['13xy', '13xyz'])
    # get data inside list of array 
    x, y = d.get(idx=idx)
    print('x shape = [{}, {}]'.format(x[0].shape, x[1].shape))
    print('y shape = [{}, {}]'.format(y[0].shape, y[1].shape))
    
    
    info = {}
    info['user_ID'] = 0 
    info['modalities'] = ['13xy', '13xyz']
    info['user_sexe'] = 'homme'
    info['user_glass'] = 0 
    info['user_ethnie'] = 'Freddie Mercury sosie'

    import faceandmouse.dataset.dataExtractor as de 
    d.add_folder('./data_to_add', info, 
                 [de.compute_13xy, de.compute_13xyz])
