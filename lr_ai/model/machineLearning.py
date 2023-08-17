"""
This is is LR Technologies Face&Mouse application.
Last Update by Adrien Dorise - April 2023

This package references all "classic" machine learniong classes used in the application.
Scikit-learn is the main API used.
Author: Adrien Dorise (adorise@lrtechnologies.fr) - LR Technologies
Created: Feb 2023
Last updated: Adrien Dorise - July 2023

"""

from sklearn.linear_model import TweedieRegressor, BayesianRidge, SGDRegressor
from sklearn import svm, neighbors, naive_bayes, tree, ensemble
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import joblib
import numpy as np
import pandas as pd


class Regressor():
    
    
    def __init__(self, model, lossMetric=metrics.mean_absolute_error, verbose=False, 
                 tweedie_param = [1, 0.5, "log"], 
                 bayes_param = [1e-05,1e-05,1e-05,1e-05], 
                 SGD_param = ['squared_error', 'l1',0.0001,'optimal'], 
                 SVM_param = ["rbf",1,0.1,0.5,'auto'], 
                 KNN_param = [3,1], 
                 tree_param = ["absolute_error",500], 
                 forest_param = ["squared_error",50,100], 
                 AdaBoost_param = [None, 50, 1, "linear"], 
                 GBoost_param = [50, 0.5, "squared_error"]):
        """
        Initialise the model with desired algorithm
        
        Args:
            model (string): algorithm selection
            lossMetric (sklearn.metrics): metrics used for loss calculation when evaluating fit or predict. It is also used for gridSearch. List on https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            verbose (bool): Put to true to print details during training. Default to False.
            *args: Parameters of each specific algorithm
        """
        self.metric = lossMetric
        self.choice = model
        self.model = []
        if model == "tweedie": #Generalized models
            self.model.append(TweedieRegressor(power=tweedie_param[0], alpha=tweedie_param[1], link=tweedie_param[2],verbose=verbose)) 
            self.model.append(TweedieRegressor(power=tweedie_param[0], alpha=tweedie_param[1], link=tweedie_param[2],verbose=verbose)) 
        elif model == "bayesLinear": #Bayesian regression
            self.model.append(BayesianRidge(alpha_1=bayes_param[0], alpha_2=bayes_param[1], lambda_1=bayes_param[2], lambda_2=bayes_param[3], verbose=verbose))
            self.model.append(BayesianRidge(alpha_1=bayes_param[0], alpha_2=bayes_param[1], lambda_1=bayes_param[2], lambda_2=bayes_param[3], verbose=verbose))
        elif model == "SGD": #Stochastic Gradient Descent
            self.model.append(SGDRegressor(loss=SGD_param[0], penalty=SGD_param[1], alpha=SGD_param[2],  learning_rate=SGD_param[3], verbose=verbose)) 
            self.model.append(SGDRegressor(loss=SGD_param[0], penalty=SGD_param[1], alpha=SGD_param[2],  learning_rate=SGD_param[3], verbose=verbose)) 
        elif model == "SVM": #Kernel support vector machines
            self.model.append(svm.SVR(kernel=SVM_param[0], C=SVM_param[1], epsilon=SVM_param[2], coef0=SVM_param[3], gamma=SVM_param[4],verbose=verbose)) 
            self.model.append(svm.SVR(kernel=SVM_param[0], C=SVM_param[1], epsilon=SVM_param[2], coef0=SVM_param[3], gamma=SVM_param[4],verbose=verbose)) 
        elif model == "KNN": #K-Nearest Neighbors
            self.model.append(neighbors.KNeighborsRegressor(n_neighbors=KNN_param[0], p=KNN_param[1]))
        elif model == "tree": #Decision Tree
            self.model.append(tree.DecisionTreeRegressor(criterion=tree_param[0], max_depth=tree_param[1]))
        elif model == "forest": #Random forest
            self.model.append(ensemble.RandomForestRegressor(criterion=forest_param[0], max_depth=forest_param[1], n_estimators=forest_param[2],verbose=verbose))
        elif model == "AdaBoost": #AdaBoost
            self.model.append(ensemble.AdaBoostRegressor(estimator=AdaBoost_param[0],n_estimators=AdaBoost_param[1], learning_rate=AdaBoost_param[2], loss=AdaBoost_param[3]))
            self.model.append(ensemble.AdaBoostRegressor(estimator=AdaBoost_param[0],n_estimators=AdaBoost_param[1], learning_rate=AdaBoost_param[2], loss=AdaBoost_param[3]))
        elif model == "GBoost": #Gradient tree boosting GBoost
            self.model.append(ensemble.GradientBoostingRegressor(n_estimators=GBoost_param[0], learning_rate=GBoost_param[1], loss=GBoost_param[2],verbose=verbose))
            self.model.append(ensemble.GradientBoostingRegressor(n_estimators=GBoost_param[0], learning_rate=GBoost_param[1], loss=GBoost_param[2],verbose=verbose))
        else:
            self.choice = "tree"
            self.model.append(tree.DecisionTreeRegressor())
        
        
    def gridSeach(self, parameters, train_set, verbose=2, parallel_jobs=-1, save_path="models/paramSearch/gridSearch"):
        """Perform a parameter search of the model using Kfold cross validation.
        The search object is then save into a joblib object as well as a csv file.
    

        Args:
            parameters (dict): Set of parameters to search
            train_set (DataLoader): data set used to search the parameters
            verbose (int, optional): Controls the verbosity between [0,3]. Defaults to 2.
            parallel_jobs (int, optional): Number of jobs in parallel. -1 means all processors. Defaults to -1.
            save_path (str, optional): Path+name to save the search. Defaults to "models/paramSearch/gridSearch".
        Return:
            Return the search object.
        """
        
        searchResults = []
        modelIter = 0
        inputs,target = self.extract_set(train_set)
        
        for mod in self.model:
            print(f"Grid search for model{modelIter} of {self.choice}")
            search = GridSearchCV(mod, parameters, verbose=verbose, n_jobs=parallel_jobs)
            if(len(self.model)==1): #Only one output or multi output model
                search.fit(inputs, target)
            else:
                search.fit(inputs, target[:,modelIter])
                
            joblib.dump(search,f"{save_path}_{self.choice}{modelIter}.sav")
            pd.DataFrame(search.cv_results_).to_csv(f"{save_path}_{self.choice}{modelIter}.csv")
            searchResults.append(search)
            modelIter+=1
        
        print("\nGrid Search best params found:")
        for res in searchResults:
            print(res.best_params_)
            
        return searchResults
        
        
        
    def fit(self, train_set):
        """Train a model on a training set
        
        Note that most of scikit learn models do not work with mini-batch training.
        For now, the whole data set is taking for the training: No epochs, no batches
        
        Args:
            train_set (torch.utils.data.DataLoader): Training set used to fit the model. This variable contains batch size information + features + target 
        """

        print(f"\nTraining <{self.choice}> START")
        batch_loss = []

        inputs,target = self.extract_set(train_set)
        
        #Training
        if(len(self.model)==1): #Only one output or multi output model
            self.model[0].fit(inputs,target)
        else:
            modelIter = 0
            for mod in self.model:
                mod.fit(inputs, target[:,modelIter])
                modelIter+=1

            
        #Get loss
        outputs = self.forward(inputs)
        loss = self.metric(target,outputs)
        print(f"Training loss is: {loss}")

        print('Finished Training')  
        
    
    def predict(self, test_set):
        """Use the trained model to predict a target values on a test set. The target must be available to calculate score.
        If no target available, use forward method.
        
        For now, we assume that the target value is known, so it is possible to calculate an error value.
        
        Args:
            test_set (torch.utils.data.DataLoader): Data set for which the model predicts a target value. This variable contains batch size information + features + target 

        Returns:
            mean_loss (float): the average error for all batch of data.
        """

        inputs,target = self.extract_set(test_set)
        # forward
        outputs = self.forward(inputs)
        score = self.metric(target,outputs)
        return score, [target.detach().numpy(), outputs]
    
    
    def forward(self, data):
        """Inference phase

        Args:
            data (array): features used to get a prediction

        Returns:
            array: prediction
        """
        results = []
        for mod in self.model:
            results.append(mod.predict(data))
        return np.array(results).reshape(len(results[0]),-1)
    
    
    def extract_set(self,dataset):
        """Extract features and targets from a dataLoader object
        Tool function used by other function of the class (fit, predict, gridSearch)
    
        Args:
            dataset (DataLoader): DataLoader obejct containing a data set

        Raises:
            Warning: Most scikit-learn models do not implement mini-batch training. Therefore, mini-batch is disable for this class. 
                If the DataLoader contain multiple batch, an error is raised.

        Returns:
            feature set (array)
            target set (array)
        """
        for i, data in enumerate(dataset, 0):
            if(dataset.batch_size != len(dataset.dataset)):
                raise Warning(f"The number of batch exceed 1. This program does not support multi-batch processing. Make sure that batch_size is equal to the number of inputs.\nBatch_size / input: [{dataset.batch_size} / {len(dataset.dataset)}]")
            else:
                # get the inputs; data is a list of [inputs, target]
                return data[0],data[1]
    
    def saveModel(self, path):
        """Save the model state in a sav file
        The model type is added to the file name
        
        If the folder specified does not exist, an error is sent
        If a file already exist, the existing file is erased

        Args:
            path (string): file path without the model type and extension
        """
        iter = 0
        for mod in self.model:
            joblib.dump(mod,f"{path}_{self.choice}{iter}.sav")
            iter+=1
    
    def loadModel(self, path):
        """Load a model from a .sav file

        Args:
            path (string): file path without the model type and extension (ex: modelFoder/name instead of modelFolder/name_KNN1.sav)
        """
        iter = 0
        for i in range(len(self.model)): #Can't use enumarator call as it creates a copy of self.model
            self.model[i] = joblib.load(f"{path}_{self.choice}{iter}.sav")
            iter+=1
            


if __name__ == "__main__":
    #!!! Parameters !!!
    from lr_ai.config.ML_config import *
    import os
    from lr_ai.features.tracker_toolbox import select_points
    import lr_ai.features.preprocessing as pr
    from lr_ai.config.ML_config import parametersModel

    trainPath = r"data/split1/train/"
    validPath = r"data/split1/val/"
    testPath = r"data/split1/test/"

    trainPath = validPath = testPath = visu_path = r"data/debug/"


    save=True
    trackerVersion = 1
    doParamSearch = False
    models = ["SVM"]
    param = parametersModel[0]
        
    
    #!!! Load data set !!!
    train_set,scaler = pr.loader(trainPath, no_batch=True, shuffle=True)
    val_set,scaler = pr.loader(testPath, no_batch=True, shuffle=False)
    
    #train_set, val_set = pr.dataSplit(features, targets, no_batch=True, split=0.8)
    
    
    #for i, (inputs, targets) in enumerate(train_set):
    #   print(f"{targets} / {inputs}")


    
    
    #!!! Parameter selection !!!
    if(doParamSearch):
        iter = 0
        for mod in models:
            userModel = Regressor(mod)
            search = userModel.gridSeach(param[iter],train_set)    
            iter += 1
        
    #!!! Training !!!
    userModel = Regressor(models[0],verbose=True)
    userModel.fit(train_set)
    if(save):
        userModel.saveModel(f"models/tmp/ML_model")
                    
    #!!! Testing !!!
    #userModel.loadModel(f"{savePath}debug{iter}")
    result, out = userModel.predict(val_set)
    print(f"Test score for <{models}> is: {result}")
          