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

def update_setting_dict(setting_dict):

	setting_dict["num_feat_maps"] = [48]*setting_dict["num_layers"]
	setting_dict["activations"] = [setting_dict["activations"][0]] *setting_dict["num_layers"]
	setting_dict["dropouts"] = [setting_dict["dropouts"][0]]*setting_dict["num_layers"]
 
	setting_dict["dropouts_fc_layers"] = [setting_dict["dropouts_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["nums_units_fc_layers"] = [setting_dict["nums_units_fc_layers"][0]]*setting_dict["num_fc_layers"]
	return

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='parser for input arguments')
	parser.add_argument('-ne', '--n_epoch', type=int, 
											help='set the number of epoch, \ndefault=30', 
											required=False,
											default=30)
	# parser.add_argument('-ns',  '--n_song', type=int, 
	# 										help='set the number of songs to train, \ndefault=300', 
	# 										required=False,
	# 										default=300)
	parser.add_argument('-tf', '--tf', help='whether cqt, stft, mfcc, \ndefault=cqt.', 
								required=False,
								default='cqt')
	parser.add_argument('-m', '--model', help='set the model, \ndefault=vgg_sequential.', 
								   		required=False, 
								   		default='vgg')
	parser.add_argument('-l', '--layers', type=int,
								 		help='set the number(s) of layers, \ndefault=[5], set like 4 5 6',
										default=5,
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
												default=8)

	parser.add_argument('-fm', '--feature_maps', type=int,
												help='set number of feature maps in convnet, \ndefault=48',
												required=False,
												default=48)

	parser.add_argument('-it', '--is_test', type=int,
												help='say if it is test \ndefault=0 (False)',
												required=False,
												default=0)

	args = parser.parse_args()

	if args.n_epoch:
		TR_CONST["num_epoch"] = args.n_epoch
	# if args.n_song:
	# 	TR_CONST["num_songs"] = args.n_song
	if args.tf:
		TR_CONST["tf_type"] = args.tf
	if args.model:
		TR_CONST["model_type"] = args.model
	if args.layers:
		TR_CONST["num_layers"] = args.layers
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
	if args.feature_maps:
		TR_CONST["num_feat_maps"] = [args.feature_maps]*TR_CONST["num_layers"]
	if args.is_test:
		is_test = bool(int(args.is_test))
	else:
		is_test = False
	if is_test:
		print '==== This is a test, to quickly check the code. ===='
	print 'Settings are \n --- num_epoch: %d\n --- model_type: %s' % \
			(TR_CONST["num_epoch"], TR_CONST["model_type"])
	print 'tf types:', TR_CONST["tf_type"]
	print ' --- num_layers: ', TR_CONST["num_layers"]
	print ' --- num_feat_maps: ', TR_CONST["num_feat_maps"]

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
	
	print '='*60
	print 'tf type: %s' % TR_CONST["tf_type"]
	print '='*60
	print "I'll take %d clips for each song." % TR_CONST["clips_per_song"]
	train_x, train_y, valid_x, valid_y, test_x, test_y = my_utils.load_all_sets_from_hdf(tf_type=TR_CONST["tf_type"],
																				n_dim=dim_latent_feature,
																				task_cla=TR_CONST['isClass'])
	# *_y is not correct - 01 Jan 2016. Use nympy files directly.
	train_y, valid_y, test_y = my_utils.load_all_labels(n_dim=dim_latent_feature, 
														num_fold=10, 
														clips_per_song=3)
	TR_CONST["height_image"] = train_x.shape[2]
	TR_CONST["width_image"]  = train_x.shape[3]

	update_setting_dict(TR_CONST) # 
	if is_test:
		pdb.set_trace()
		train_x = train_x[0:24]
		train_y = train_y[0:24]
		valid_x = valid_x[0:24]
		valid_y = valid_y[0:24]
		test_x = test_x[0:24]
		test_y = test_y[0:24]
		
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	# train_x : (num_samples, num_channel, height, width)	
	hyperparams_manager = my_utils.Hyperparams_Manager()
	model_name = hyperparams_manager.get_name(TR_CONST)
	if is_test:
		mode_name = 'test_' + model_name
	else:
		model_name = time.strftime('%m-%d-%Hh_') + model_name
	hyperparams_manager.save_new_setting(TR_CONST)
	print '-'*60
	print 'model name: %s' % model_name
	print '-'*60
	model_name_dir = model_name + '/'
	model_weight_name_dir = 'w_' + model_name + '/'
	fileout = model_name + '_results'
	
	if not os.path.exists(PATH_RESULTS + model_name_dir):
		os.mkdir(PATH_RESULTS + model_name_dir)
		os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
		os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
		os.mkdir(PATH_RESULTS + model_weight_name_dir)
	start = time.time()

	print "--- going to build a keras model with height:%d, width:%d, num_labels:%d" \
							% (train_x.shape[2], train_x.shape[3], train_y.shape[1])

	my_utils.write_setting_as_texts(PATH_RESULTS + model_name_dir, TR_CONST)
 	if TR_CONST["isRegre"]:
 		print '--- ps. this is a regression task. ---'
		model = my_keras_models.build_regression_convnet_model(setting_dict=TR_CONST, is_test=is_test)

	else:
		print '--- ps. this is a classification task. ---'
		print 'Hey, dont classify this.'
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
	
	patience = 5

	if TR_CONST["isRegre"]:
		#history = my_keras_utils.History_Regression_Val()
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
														patience=patience, 
														verbose=0)
	else:
		h#istory = my_keras_utils.History_Classification_Val()
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
														patience=patience, 
														verbose=0)
	#train!
	my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
										filename_prefix='INIT_', 
										normalize='local', 
										mono=True)

	if TR_CONST["tf_type"] == 'cqt':
		batch_size = 32
	elif TR_CONST["tf_type"] == 'stft':
		batch_size = 16
	elif TR_CONST["tf_type"] == 'mfcc':
		batch_size = 48
	else:
		raise RuntimeError('batch size for this? %s' % TF_CONST["tf_type"])

	predicted = model.predict(test_x, batch_size=batch_size)
	
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])

	keras_plot(model, to_file=PATH_RESULTS + model_name_dir + 'images/'+'graph_of_model.png')
	print '--- train starts ---'
	if TR_CONST["isRegre"]:
		if is_test:
			history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
												batch_size=batch_size, 
												nb_epoch=TR_CONST["num_epoch"], 
												show_accuracy=False, 
												verbose=1, 
												callbacks=[weight_image_saver],
												shuffle=False)
		else:
			history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
												batch_size=batch_size, 
												nb_epoch=TR_CONST["num_epoch"], 
												show_accuracy=False, 
												verbose=1, 
												callbacks=[weight_image_saver, early_stopping, checkpointer],
												shuffle=False)
		loss_testset = model.evaluate(test_x, test_y, show_accuracy=False, batch_size=batch_size)
	else:
		batch_size = batch_size / 2
		history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
									batch_size=batch_size, 
									nb_epoch=TR_CONST["num_epoch"], 
									show_accuracy=True, 
									verbose=1, 
									callbacks=[early_stopping, weight_image_saver, checkpointer],
									shuffle=False)
		loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
	
	predicted = model.predict(test_x, batch_size=batch_size)
	#save results
	model.save_weights(PATH_RESULTS + model_weight_name_dir + ('final_after_%d.keras' % TR_CONST["num_epoch"]), overwrite=True) 
	np.save(PATH_RESULTS + model_name_dir + fileout + '_history.npy', [history.history['loss'], history.history['val_loss']])
	np.save(PATH_RESULTS + model_name_dir + fileout + '_loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])

	if TR_CONST["isRegre"]:
		
		my_plots.export_history(history.history['loss'], history.history['val_loss'],
												acc=None, 
												val_acc=None, 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	else:
		my_plots.export_history(history.history['loss'], history.history['val_loss'], 
												acc=history.history['acc'], 
												val_acc=history.history['val_acc'], 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	
	my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
										filename_prefix='', 
										normalize='local', 
										mono=True)
	pdb.set_trace()
	min_loss = np.min(history.history['val_loss'])
	arg_min = np.argmin(history.history['val_loss'])+1
	best_batch = history.history['batch'][arg_min]
	num_run_epoch = history.history['batch'][-1]
	os.mkdir(PATH_RESULTS + model_name + '_%s_%06.4f_at_%d_of_%d' % (TR_CONST["loss_function"], min_loss, best_batch, num_run_epoch))
	