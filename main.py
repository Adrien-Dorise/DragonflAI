'''
 Main file to launch a debug test of DragonflAI
 DragonflAI evolves around 
 Author: Adrien Dorise (adrien.dorise@hotmail.com)
 Created: June 2023
 Last updated: Adrien Dorise - August 2023
'''

if __name__ == "__main__":
    
	import dragonflai.config.ML_config as ML_config
	import dragonflai.config.NN_config as NN_config
	import dragonflai.config.data_config as data_config
	from dragonflai.experiment import *
	from dragonflai.dataset.create_blob_sets import create_dataset, plot_dataset
	from dragonflai.preprocess import data_loader
	from dragonflai.utils.utils_model import modelType, taskType

	features_train, targets_train = create_dataset(list_N=data_config.list_N, list_center=data_config.list_centroid)
	train_set, validation_set = data_loader.create_loaders(features=features_train, targets=targets_train, batch_size=NN_config.batch_size)
	features_test, targets_test = create_dataset(list_N=data_config.list_N, list_center=data_config.list_centroid, seed=1331)
	test_set, visualisation_set = data_loader.create_loaders(features=features_test, targets=targets_test, test_size=0.05, batch_size=NN_config.batch_size)

	plot_dataset(features_train,targets_train,f"{data_config.save_path}train_datatset")
	
	experiment = Experiment(model = NN_config.NN_model,
                         train_set = train_set,
                         validation_set = validation_set,
                         test_set = test_set,
                         visualisation_set = visualisation_set,
                         model_type = modelType.NEURAL_NETWORK,
                         task_type = taskType.CLASSIFICATION,
                         criterion = NN_config.crit,
                         n_epochs = NN_config.num_epoch,
                         batch_size = NN_config.batch_size,
                         save_path = data_config.save_path
                         )

	experiment.fit()
	experiment.predict()
	experiment.visualise()
	experiment.save(f"{data_config.save_path}experiment")
	experiment = Experiment.load(f"{data_config.save_path}experiment")
	experiment.visualise()
