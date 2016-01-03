""" To predict tags! using ilm10k data, stft or cqt representation, 
same as main_learn_tag but it load data from hdf file, 28 Dec 2015
"""
#import matplotlib
#matplotlib.use('Agg')
from constants import *
from environments import *
from training_settings import *
import argparse
import os
import pdb

import numpy as np

import keras
import my_keras_models
import my_keras_utils
from keras.utils.visualize_util import plot as keras_plot
import my_utils
import cPickle as cP
import time
import sys
import my_plots

def print_usage_and_die():
	print 'python filename num_of_epoch(int) num_of_train_song(int) tf_type model_type num_of_layers'
	print 'ex) $ python main_learn_tag.py 200 5000 cqt vgg classification 4 5 6'
	sys.exit()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='parser for input arguments')
	parser.add_argument('-ne', '--n_epoch', type=int, 
											help='set the number of epoch, \ndefault=50', 
											required=False,
											default=50)
	parser.add_argument('-ns',  '--n_song', type=int, 
											help='set the number of songs to train, \ndefault=300', 
											required=False,
											default=300)
	parser.add_argument('-tf', '--tf', help='whether cqt or stft, \ndefault=cqt.', 
								required=False,
								nargs='+',
								default=['cqt'])
	parser.add_argument('-m', '--model', help='set the model, \ndefault=vgg_sequential.', 
								   		required=False, 
								   		default='vgg')
	parser.add_argument('-l', '--layers', type=int,
								 		help='set the number(s) of layers, \ndefault=[5], set like 4 5 6',
										nargs='+',
										default=[5],
										required=False)
	parser.add_argument('-t', '--task', help='classification or regression, \ndefault=regre', 
									   required=False, 
									   default='regre')
	parser.add_argument('-cps', '--clips_per_song', type=int,
													help='set #clips/song, \ndefault=3',
													required=False,
													default=3)
	parser.add_argument('-dl', '--dim_labels', type=int,
												help='set dimension of label, \ndefault=3',
												required=False,
												default=3)
	
	parser.add_argument('-it', '--is_test', type=int,
												help='say if it is test \ndefault=0 (False)',
												required=False,
												default=0)


	args = parser.parse_args()

	if args.n_epoch:
		TR_CONST["num_epoch"] = args.n_epoch
	if args.n_song:
		TR_CONST["num_songs"] = args.n_song
	if args.tf:
		tf_types = args.tf
	if args.model:
		TR_CONST["model_type"] = args.model
	if args.layers:
		num_of_layers = args.layers
	if args.task:
		if args.task in['class', 'cla', 'c', 'classification']:
			TR_CONST["isClass"] = True
			TR_CONST["isRegre"] = False
		else:
			TR_CONST["isClass"] = False
			TR_CONST["isRegre"] = True
	if args.clips_per_song:
		TR_CONST["clips_per_song"] = args.clips_per_song
	if args.dim_labels:
		TR_CONST["dim_labels"] = args.dim_labels
	if args.is_test:
		is_test = bool(int(args.is_test))
	else:
		is_test = False
	if is_test:
		print '==== This is a test, to quickly check the code. ===='
	print 'Settings are \n --- num_epoch: %d\n --- num_songs: %d\n --- model_type: %s' % \
			(TR_CONST["num_epoch"], TR_CONST["num_songs"], TR_CONST["model_type"])
	print 'tf types:', tf_types
	print ' --- num_layers: ', TR_CONST["num_layers"]
	print ' --- task: %s' % args.task

	# label matrix
	dim_latent_feature = TR_CONST["dim_labels"]
	# label_matrix_filename = (FILE_DICT["mood_latent_matrix"] % dim_latent_feature)
	label_matrix_filename = (FILE_DICT["mood_latent_tfidf_matrix"] % dim_latent_feature) # tfidf is better!
	
	if os.path.exists(PATH_DATA + label_matrix_filename):
		label_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
	else:
		"print let's cook the mood-latent feature matrix"
		import main_prepare
		mood_tags_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
		label_matrix = main_prepare.get_LDA(X=mood_tags_matrix, 
											num_components=k, 
											show_topics=False)
		np.save(PATH_DATA + label_matrix_filename, W)
	print 'size of mood tag matrix:'
	print label_matrix.shape

	# load dataset
	for tf_type in tf_types:
		TR_CONST["tf_type"] = tf_type
		print '='*60
		print 'tf type: %s' % tf_type
		print '='*60
		print "I'll take %d clips for each song." % TR_CONST["clips_per_song"]
		#train_x, train_y, valid_x, valid_y, test_x, test_y = my_utils.load_all_sets(label_matrix=label_matrix, 
		#																	clips_per_song=TR_CONST["clips_per_song"], 
		#																	num_train_songs=TR_CONST["num_songs"], 
		#																	tf_type=TR_CONST["tf_type"])
		train_x, train_y, valid_x, valid_y, test_x, test_y = my_utils.load_all_sets_from_hdf(tf_type=TR_CONST["tf_type"],
																					n_dim=dim_latent_feature,
																					task_cla=TR_CONST['isClass'])
		# *_y is not correct - 01 Jan 2016. Use nympy files directly.

		train_y, valid_y, test_y = my_utils.load_all_labels(n_dim=dim_latent_feature, 
															num_fold=10, 
															clips_per_song=3)
		
		if is_test:
			pdb.set_trace()
			train_x = train_x[0:64]
			train_y = train_y[0:64]
			valid_x = valid_x[0:64]
			valid_y = valid_y[0:64]
			test_x = test_x[0:64]
			test_y = test_y[0:64]
			
			
		moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
		# train_x : (num_samples, num_channel, height, width)
		# learning_id =  str(np.random.randint(999999))
		
		# if TR_CONST["isClass"]:
		# 	print 'labels of train_y ratio: ' + '%4.2f, '*TR_CONST["dim_labels"] % \
		# 			tuple(np.asarray(np.sum(train_y, axis=0) / float(np.sum(train_y))))
		# 	print 'labels of valid_y ratio: '+ '%4.2f, '*TR_CONST["dim_labels"] % \
		# 			tuple(np.asarray(np.sum(valid_y, axis=0) / float(np.sum(valid_y))))
		# 	print 'labels of test_y ratio: '+ '%4.2f, '*TR_CONST["dim_labels"] % \
		# 			tuple(np.asarray(np.sum(test_y, axis=0) / float(np.sum(test_y))))
		
		for num_layers in num_of_layers:
			TR_CONST["num_layers"] = num_layers
			hyperparams_manager = my_utils.Hyperparams_Manager()
			model_name = hyperparams_manager.get_name(TR_CONST)
			if is_test:
				mode_name = 'test_' + model_name
			hyperparams_manager.save_new_setting(TR_CONST)
			print '-'*60
			print 'model name: %s' % model_name
			print '-'*60
			model_name_dir = model_name + '/'
			model_weight_name_dir = model_name + '_weights/'
			fileout = model_name + '_results'
			
			if not os.path.exists(PATH_RESULTS + model_name_dir):
				os.mkdir(PATH_RESULTS + model_name_dir)
				os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
				os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
				os.mkdir(PATH_RESULTS + model_weight_name_dir)
			my_utils.write_setting_as_texts(PATH_RESULTS + model_name_dir, TR_CONST)
			start = time.time()

			print "--- going to build a keras model with height:%d, width:%d, num_labels:%d" \
									% (train_x.shape[2], train_x.shape[3], train_y.shape[1])
		 	if TR_CONST["isRegre"]:
		 		print '--- ps. this is a regression task. ---'
		 		model = my_keras_models.build_regression_convnet_model(height=train_x.shape[2], 
		 																width=train_x.shape[3], 
		 																dropouts=TR_CONST["dropouts"]
		 																num_labels=train_y.shape[1], 
		 																num_layers=TR_CONST["num_layers"], 
		 																model_type=TR_CONST["model_type"], 
		 																num_channels=1)
			else:
				print '--- ps. this is a classification task. ---'
				model = my_keras_models.build_classification_convnet_model(height=train_x.shape[2], 
																			width=train_x.shape[3], 
																			num_labels=train_y.shape[1], 
																			num_layers=TR_CONST["num_layers"], 
																			model_type=TR_CONST["model_type"],
																			num_channels=1)		
		 	until = time.time()
		 	print "--- keras model was built, took %d seconds ---" % (until-start)
			#prepare callbacks
			checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_RESULTS + model_weight_name_dir + "weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
															verbose=1, 
															save_best_only=False)
			weight_image_saver = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
			
			if TR_CONST["isRegre"]:
				history = my_keras_utils.History_Regression_Val()
				early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
																patience=5, 
																verbose=0)
			else:
				history = my_keras_utils.History_Classification_Val()
				early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
																patience=5, 
																verbose=0)
			#train!
			my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
												filename_prefix='INIT_', 
												normalize='local', 
												mono=True)

			predicted = model.predict(train_x, batch_size=12)
			
			np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(train_y)], train_y[:len(train_y)]])

			if TR_CONST["tf_type"] == 'cqt':
				batch_size = 32
			elif TR_CONST["tf_type"] == 'stft':
				batch_size = 12
			else:
				raise RuntimeError('batch size for this? %s' % TF_CONST["tf_type"])
			keras_plot(model, to_file=PATH_RESULTS + model_name_dir + 'images/'+'graph_of_model.png')
			print '--- train starts ---'
			if TR_CONST["isRegre"]:
				model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
											batch_size=batch_size, 
											nb_epoch=TR_CONST["num_epoch"], 
											show_accuracy=False, 
											verbose=1, 
											callbacks=[history, early_stopping, weight_image_saver, checkpointer],
											shuffle=False)
				loss_testset = model.evaluate(test_x, test_y, show_accuracy=False)
			else:
				batch_size = batch_size / 2
				model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
											batch_size=batch_size, 
											nb_epoch=TR_CONST["num_epoch"], 
											show_accuracy=True, 
											verbose=1, 
											callbacks=[history, early_stopping, weight_image_saver, checkpointer],
											shuffle=False)
				loss_testset = model.evaluate(test_x, test_y, show_accuracy=True)
			
			predicted = model.predict(test_x, batch_size=12)
			#save results
			model.save_weights(PATH_RESULTS + model_weight_name_dir + ('final_after_%d.keras' % TR_CONST["num_epoch"]), overwrite=True) 
			
			np.save(PATH_RESULTS + fileout + '_history.npy', history.val_losses)
			np.save(PATH_RESULTS + fileout + '_loss_testset.npy', loss_testset)
			np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(train_y)], train_y[:len(train_y)]])
			if TR_CONST["isRegre"]:
				my_plots.export_history(history.losses, history.val_losses, 
														acc=None, 
														val_acc=None, 
														out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
			else:
				my_plots.export_history(history.losses, history.val_losses, 
														acc=history.accs, 
														val_acc=history.val_accs, 
														out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
			my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
												filename_prefix='', 
												normalize='local', 
												mono=True)
			
	# figure_filepath = PATH_FIGURE + model_name + '_history.png'
