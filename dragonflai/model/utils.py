"""
This package references all utils neural network classes used in the application.
Author: Adrien Dorise - Edouard Villain ({adorise, evillain}@lrtechnologies.fr) - LR Technologies
Created: April 2024
Last updated: Edouard Villain - April 2024
"""

import os 

from enum import Enum

class taskType(Enum):
    REGRESSION     = 1
    CLASSIFICATION = 2
    SEGMENTATION   = 3
    
class dragonflAIProgressBar():
    '''Custom progress bar'''
    def __init__(self, mode=taskType.REGRESSION, verbosity=1, epochs=0, steps_per_epoch_train=0, steps_per_epoch_test=0):
        self.mode                  = mode
        self.verbosity             = verbosity
        self.current_batch_train   = 0
        self.current_batch_test    = 0
        self.epoch                 = 0
        self.epochs                = epochs
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_test  = steps_per_epoch_test
        self.duration_t            = 0
        self.lr                    = 0
        
    def set_current_batch_train(self, current_batch_train):
        '''set current batch train'''
        self.current_batch_train = current_batch_train
        
    def set_current_batch_test(self, current_batch_test):
        '''set current batch test'''
        self.current_batch_test = current_batch_test
        
    def set_lr(self, lr):
        '''set current learning rate'''
        self.lr = lr 
    
    def set_epoch(self, epoch):
        '''set current epoch'''
        self.epoch = epoch 
    
    def set_isTrain(self, istrain):
        '''set current status, training or not'''
        self.istrain = istrain 
        
    def set_loss(self, loss, val_loss):
        '''set current loss and validation loss'''
        self.loss = loss 
        self.val_loss = val_loss 
        
    def set_accuracy(self, acc, val_acc):
        '''set current accuracy and validation accuracy'''
        self.acc = acc 
        self.val_acc = val_acc 
        
    def set_duration(self, duration_t):
        '''set current duration'''
        self.duration_t = duration_t 
        
        
    def plot_log(self, *args, **kwargs):
        '''plot log during training'''
        if self.verbosity > 0:
            column, _ = os.get_terminal_size()
            verbose = self.verbosity
            
            if self.current_batch_test == self.steps_per_epoch_test:
                verbose = 2
                
            if not self.istrain:
                self.lr = 0
                    
            size_bar = column - 122
            i        = (size_bar * (self.current_batch_train + self.current_batch_test) // (self.steps_per_epoch_train + self.steps_per_epoch_test))
            end      = '\r'
            total    = (self.steps_per_epoch_train + self.steps_per_epoch_test) 
            if self.current_batch_test == self.steps_per_epoch_test:
                est_t = 'time used = {} s.'.format(self.duration_t)
            else:
                est_t = 'time left ~ {} s.'.format(self.duration_t)
            
            if verbose == 2:
                end = '\n'
            if self.mode in [taskType.REGRESSION, taskType.SEGMENTATION]:
                print('[{:4d}/{:4d}, {:4d}/{:4d}, {:4d}/{:4d}] : [{}>{}] : lr = {:.3e} - loss = {:.3e} - val = {:.3e} - {}  '.format(self.epoch,self.epochs,
                                                                                            self.current_batch_train, self.steps_per_epoch_train,
                                                                                            self.current_batch_test, self.steps_per_epoch_test,
                                                                                            '=' * i, ' ' * (size_bar - i),
                                                                                            self.lr, self.loss, self.val_loss, est_t), end=end)
            if self.mode == taskType.CLASSIFICATION:
                print('[{:4d}/{:4d}, {:4d}/{:4d}, {:4d}/{:4d}] : [{}>{}] : lr = {:.3e} - acc = {:.3e} - val = {:.3e} - {}  '.format(self.epoch,self.epochs,
                                                                                            self.current_batch_train, self.steps_per_epoch_train,
                                                                                            self.current_batch_test, self.steps_per_epoch_test,
                                                                                            '=' * i, ' ' * (size_bar - i),
                                                                                            self.lr, self.loss, self.val_loss, est_t), end=end)