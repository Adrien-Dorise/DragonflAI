'''
 Main file to launch a debug test of the LR Technologies artificial intelligence template
 Author: Adrien Dorise (adrien.dorise@hotmail.com)
 Created: June 2023
 Last updated: Adrien Dorise - August 2023
'''



if __name__ == "__main__":
    
	from dragonflai.config.ML_config import *
	from dragonflai.config.NN_config import *
	from dragonflai.config.data_config import *
	from dragonflai.experiment import *
	
	experiment = Experiment(NN_model,
			train_path,
			val_path,
			test_path,
			visu_path,
			classification = classification,
			input_type = input_type,
			crop = crop,
			coords = coords,
			tracker_version = tracker_version,
			sequence_length = seq_length,
			num_epoch = num_epoch,
			batch_size = batch_size,
			learning_rate = lr,
			weight_decay = wd,
			optimizer = optimizer,
			criterion = crit,
			scaler = scaler,
			nb_workers = nb_workers)

	experiment.fit()
	experiment.predict()
	experiment.visualise_lookAtScreen()
