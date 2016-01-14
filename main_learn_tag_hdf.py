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

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

def update_setting_dict(setting_dict):

	setting_dict["num_feat_maps"] = [setting_dict["num_feat_maps"][0]]*setting_dict["num_layers"]
	setting_dict["activations"] = [setting_dict["activations"][0]] *setting_dict["num_layers"]
	setting_dict["dropouts"] = [setting_dict["dropouts"][0]]*setting_dict["num_layers"]
	setting_dict["regulariser"] = [setting_dict["regulariser"][0]]*setting_dict["num_layers"]
	# setting_dict["regulariser"][0] = ('l1', setting_dict["regulariser"][0][1]* 9) # bigger regulariser 
	# setting_dict["regulariser"][1] = ('l1', setting_dict["regulariser"][1][1]* 3)

	# tweak
	# setting_dict["dropouts"] = [0.25]*2 + [0.0]*(setting_dict["num_layers"]-2)
	# setting_dict["dropouts"] = [0.25]*(setting_dict["num_layers"])
	# setting_dict["regulariser"] = [('l1', 5e-5), ('l1',1e-4)] + [setting_dict["regulariser"][0]]*(setting_dict["num_layers"]-2)
	# setting_dict["regulariser"] = [None]*(setting_dict["num_layers"])
	# setting_dict["!memo"] = setting_dict["!memo"] + '_hybrid_dropout_and_l2'

	setting_dict["dropouts_fc_layers"] = [setting_dict["dropouts_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["nums_units_fc_layers"] = [setting_dict["nums_units_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["activations_fc_layers"] = [setting_dict["activations_fc_layers"][0]]*setting_dict["num_fc_layers"]
	setting_dict["regulariser_fc_layers"] = [setting_dict["regulariser_fc_layers"][0]]*setting_dict["num_fc_layers"]

	#tweak 2
	# setting_dict["regulariser"] = [None]*(setting_dict["num_layers"])
	# setting_dict["regulariser_fc_layers"] = [None]*(setting_dict["num_fc_layers"])
	
	return

def run_with_setting(hyperparams, argv):
	print '#'*60
	#function: input args: TR_CONST, sys.argv.
	# -------------------------------
	if hyperparams["is_test"]:
		print '==== This is a test, to quickly check the code. ===='
		print 'excuted by $ ' + ' '.join(argv)
	
	# label matrix
	dim_latent_feature = hyperparams["dim_labels"]
	# label_matrix_filename = (FILE_DICT["mood_latent_matrix"] % dim_latent_feature)
	label_matrix_filename = (FILE_DICT["mood_latent_tfidf_matrix"] % dim_latent_feature) # tfidf is better!
	
	if os.path.exists(PATH_DATA + label_matrix_filename):
		label_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
	else:
		"print let's create a new mood-latent feature matrix"
		import main_prepare
		mood_tags_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
		label_matrix = main_prepare.get_LDA(X=mood_tags_matrix, 
											num_components=k, 
											show_topics=False)
		np.save(PATH_DATA + label_matrix_filename, W)
	# print 'size of mood tag matrix:'
	# print label_matrix.shape

	# load dataset
	
	train_x, valid_x, test_x, = my_utils.load_all_sets_from_hdf(tf_type=hyperparams["tf_type"],
																				n_dim=dim_latent_feature,
																				task_cla=hyperparams['isClass'])
	# *_y is not correct - 01 Jan 2016. Use nympy files directly.
	train_y, valid_y, test_y = my_utils.load_all_labels(n_dim=dim_latent_feature, 
														num_fold=10, 
														clips_per_song=3)
	hyperparams["height_image"] = train_x.shape[2]
	hyperparams["width_image"]  = train_x.shape[3]
	if hyperparams["debug"]:
		pdb.set_trace()
	if hyperparams["is_test"]:
		# train_x = train_x[0:24]
		# train_y = train_y[0:24]
		# valid_x = valid_x[0:24]
		# valid_y = valid_y[0:24]
		# test_x = test_x[0:24]
		# test_y = test_y[0:24]
		train_y = train_y[:,[0]]
		valid_y = valid_y[:,[0]]
		test_y  = test_y[:,[0]]
		hyperparams["dim_labels"] = 1
		print 'Output is one dimensional value.'
	

	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	# train_x : (num_samples, num_channel, height, width)	
	hyperparams_manager = my_utils.Hyperparams_Manager()
	nickname = hyperparams_manager.get_name(hyperparams)
	timename = time.strftime('%m-%d-%Hh%M')
	if hyperparams["is_test"]:
		model_name = 'test_' + nickname
	else:
		model_name = timename + '_' + nickname
	hyperparams_manager.save_new_setting(hyperparams)
	print '-'*60
	print 'model name: %s' % model_name
	model_name_dir = model_name + '/'
	model_weight_name_dir = 'w_' + model_name + '/'
	fileout = model_name + '_results'
	
	if not os.path.exists(PATH_RESULTS + model_name_dir):
		os.mkdir(PATH_RESULTS + model_name_dir)
		os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
		os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
		os.mkdir(PATH_RESULTS_W + model_weight_name_dir)
	start = time.time()

	hyperparams_manager.write_setting_as_texts(PATH_RESULTS + model_name_dir, hyperparams)
 	hyperparams_manager.print_setting(hyperparams)
 	if hyperparams["isRegre"]:
 		
		model = my_keras_models.build_regression_convnet_model(setting_dict=hyperparams)

	else:
		print '--- ps. this is a classification task. ---'
		print 'Hey, dont classify this.'
		model = my_keras_models.build_classification_convnet_model(height=train_x.shape[2], 
																	width=train_x.shape[3], 
																	num_labels=train_y.shape[1], 
																	num_layers=hyperparams["num_layers"], 
																	model_type=hyperparams["model_type"],
																	num_channels=1)		
 	until = time.time()
 	print "--- keras model was built, took %d seconds ---" % (until-start)
 	keras_plot(model, to_file=PATH_RESULTS + model_name_dir + 'images/'+'graph_of_model_'+hyperparams["!memo"]+'.png')
	#prepare callbacks
	checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5", 
													verbose=1, 
								             		save_best_only=True)
	weight_image_monitor = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
	patience = 3
	if hyperparams["is_test"] is True:
		patience = 99999999
	if hyperparams["isRegre"]:
		#history = my_keras_utils.History_Regression_Val()
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
														patience=patience, 
														verbose=0)
	else:
		#history = my_keras_utils.History_Classification_Val()
		early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
														patience=patience, 
														verbose=0)
	#save image of weights
	my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
										filename_prefix='INIT_', 
										normalize='local', 
										mono=True)
	# other constants
	if hyperparams["tf_type"] == 'cqt':
		batch_size = 24
	elif hyperparams["tf_type"] == 'stft':
		batch_size = 12
	elif hyperparams["tf_type"] == 'mfcc':
		batch_size = 48
	else:
		raise RuntimeError('batch size for this? %s' % hyperparams["tf_type"])
	if hyperparams['model_type'] == 'vgg_original':
		batch_size = (batch_size * 3)/5

	predicted = model.predict(test_x, batch_size=batch_size)
	if hyperparams['debug'] == True:
		pdb.set_trace()
	print 'mean of target value:'
	print np.mean(test_y, axis=0)
	print 'mean of predicted value:'
	print np.mean(predicted, axis=0)
	print 'mse with just predicting average is %f' % np.mean((test_y - np.mean(test_y, axis=0))**2)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	#train!
	
	print '--- train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % hyperparams["num_epoch"]
	f = open('will_stop.keunwoo', 'w')
	f.close()
	total_history = {}
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	if hyperparams['is_test']:
		callbacks = [weight_image_monitor]
	else:
		callbacks = [weight_image_monitor, early_stopping, checkpointer]

	while True:
		if hyperparams["isRegre"]:
			history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
													batch_size=batch_size, 
													nb_epoch=num_epoch, 
													show_accuracy=False, 
													verbose=1, 
													callbacks=callbacks,
													shuffle='batch')
		else:
			batch_size = batch_size / 2
			history=model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
										batch_size=batch_size, 
										nb_epoch=num_epoch, 
										show_accuracy=True, 
										verbose=1, 
										callbacks=callbacks,
										shuffle='batch')
		total_epoch += num_epoch
		print '%d-th epoch is complete' % total_epoch
		my_utils.append_history(total_history, history.history)
		if os.path.exists('max_epoch.npy'):
			max_epoch = np.load('max_epoch.npy')
			if total_epoch < max_epoch:
				num_epoch = max_epoch - total_epoch
				f = open('will_stop.keunwoo', 'w')
				f.close()
				#add a line to remove npy file.
				continue
		if os.path.exists('will_stop.keunwoo'):
			if hyperparams["isRegre"]:
				loss_testset = model.evaluate(test_x, test_y, show_accuracy=False, batch_size=batch_size)
			else:
				loss_testset = model.evaluate(test_x, test_y, show_accuracy=True, batch_size=batch_size)
			break
		else:
			num_epoch = 1
			print ' *** will go for another one epoch. '
			print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	#
	best_batch = np.argmin(total_history['val_loss'])+1
	# model.load_weights() # load the best model
	predicted = model.predict(test_x, batch_size=batch_size)
	print predicted[:10]

	if hyperparams["debug"] == True:

		pdb.set_trace()
	if not hyperparams['is_test']:
		model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 
	predicted = model.predict(test_x, batch_size=batch_size)
	print predicted[:10]
	#save results
	np.save(PATH_RESULTS + model_name_dir + fileout + '_history.npy', [total_history['loss'], total_history['val_loss']])
	np.save(PATH_RESULTS + model_name_dir + fileout + '_loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code
	if hyperparams["isRegre"]:
		my_plots.export_history(total_history['loss'], total_history['val_loss'],
												acc=None, 
												val_acc=None, 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	else:
		my_plots.export_history(total_history['loss'], total_history['val_loss'], 
												acc=total_history['acc'], 
												val_acc=total_history['val_acc'], 
												out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	
	# my_plots.save_model_as_image(model, save_path=PATH_RESULTS + model_name_dir + 'images/', 
	# 									filename_prefix='', 
	# 									normalize='local', 
	# 									mono=True)
	
	
	min_loss = np.min(total_history['val_loss'])
	best_batch = np.argmin(total_history['val_loss'])+1
	num_run_epoch = len(total_history['val_loss'])
	oneline_result = '%6.4f, %d_of_%d, %s' % (min_loss, best_batch, num_run_epoch, model_name)
	with open(PATH_RESULTS + model_name_dir + oneline_result, 'w') as f:
		pass
	f = open( (PATH_RESULTS + '%s_%s_%06.4f_at_(%d_of_%d)_%s'  % \
		(timename, hyperparams["loss_function"], min_loss, best_batch, num_run_epoch, nickname)), 'w')
	f.close()
	with open('one_line_log.txt', 'a') as f:
		f.write('%6.4f, %d/%d, %s' % (min_loss, best_batch, num_run_epoch, model_name))
		f.write(' ' + ' '.join(argv) + '\n')
	print '========== DONE: %s ==========' % model_name
	return min_loss

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='parser for input arguments')
	parser.add_argument('-ne', '--n_epoch', type=int, 
											help='set the number of epoch, \ndefault=30', 
											required=False)
	parser.add_argument('-tf', '--tf', help='whether cqt, stft, mfcc, \ndefault=cqt.', 
								required=False)
	parser.add_argument('-m', '--model', help='set the model, \ndefault=vgg_sequential.', 
								   		required=False)
	parser.add_argument('-l', '--layers', type=int,
								 		help='set the number(s) of layers, \ndefault=[5], set like 4, 5, 6',
										required=False)
	parser.add_argument('-lfc', '--num_fc_layers', type=int,
								 		help='set the number(s) of fc layers, \ndefault=[2], set like 1, 2, 3',
										required=False)
	parser.add_argument('-t', '--task', help='classification or regression, \ndefault=regre', 
									   required=False)
	parser.add_argument('-op', '--optimiser', help='optimiser - rmsprop, sgd, adagrad, adam, adadelta \ndefault=rmsprop', 
									   required=False)
	parser.add_argument('-lf', '--loss_function', help='loss function - binary_crossentropy, rmse\ndefault=binary_crossentropy', 
									   required=False)
	parser.add_argument('-act', '--activations', help='activations - relu, lrelu, prelu, elu \ndefault=relu', 
									   required=False)
	parser.add_argument('-cps', '--clips_per_song', type=int,
													help='set #clips/song, \ndefault=3',
													required=False)
	parser.add_argument('-dl', '--dim_labels', type=int,
												help='set dimension of label, \ndefault=3',
												required=False)
	parser.add_argument('-fm', '--feature_maps', type=int,
												help='set number of feature maps in convnet, \ndefault=48',
												required=False)
	parser.add_argument('-nu', '--number_units', type=int,
												help='set number of units in fc layers, \ndefault=512',
												required=False)	
	parser.add_argument('-it', '--is_test', type=int,
												help='say if it is test \ndefault=0 (False)',
												required=False)
	parser.add_argument('-memo', '--memo', 	help='short memo \ndefault=""',
											required=False)
	parser.add_argument('-do', '--dropout', type=float,
											help='dropout value that is applied to conv',
											required=False)
	parser.add_argument('-do_fc', '--dropout_fc', type=float,
												help='dropout value that is applied to FC layers',
												required=False)
	parser.add_argument('-reg', '--regulariser', type=float,
												help='regularise coeff that is applied to conv',
												required=False)
	parser.add_argument('-reg_fc', '--regulariser_fc', type=float,
														help='regularise coeff that is applied to fc layer',
														required=False)
	parser.add_argument('-bn', '--batch_normalization', type=str,
														help='BN for conv layers',
														required=False)
	parser.add_argument('-bn_fc', '--batch_normalization_fc', type=str,
															help='BN for fc layers',
															required=False)
	parser.add_argument('-debug', '--debug', type=str,
											help='if debug',
											required=False)
	parser.add_argument('-lr', '--learning_rate', type=float,
													help='learning_rate',
													required=False)
	parser.add_argument('-ol', '--output_layer', type=str,
												help='sigmoid, linear',
												required=False )
	
	
	args = parser.parse_args()
	
	if args.layers:
		TR_CONST["num_layers"] = args.layers
	if args.num_fc_layers:
		TR_CONST["num_fc_layers"] = args.num_fc_layers
	if args.n_epoch:
		TR_CONST["num_epoch"] = args.n_epoch
	if args.tf:
		TR_CONST["tf_type"] = args.tf
		print 'tf-representation type is input by: %s' % TR_CONST["tf_type"]
	if args.optimiser:
		TR_CONST["optimiser"] = args.optimiser
	if args.loss_function:
		TR_CONST["loss_function"] = args.loss_function
	if args.model:
		TR_CONST["model_type"] = args.model
	if args.activations:
		TR_CONST["activations"] = [args.activations] * TR_CONST["num_layers"]
		TR_CONST["activations_fc_layers"] = [args.activations] * TR_CONST["num_fc_layers"]
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
	if args.number_units:
		TR_CONST["nums_units_fc_layers"] = [args.number_units]*TR_CONST["num_fc_layers"]
	if args.is_test:
		TR_CONST["is_test"] = bool(int(args.is_test))
	if args.memo:
		TR_CONST["!memo"] = args.memo
	else:
		TR_CONST["!memo"] = ''
	if args.dropout or args.dropout == 0.0:
		TR_CONST["dropouts"] = [args.dropout]*TR_CONST["num_layers"]
	if args.dropout_fc or args.dropout_fc == 0.0:
		TR_CONST["dropouts_fc_layers"] = [args.dropout_fc]*TR_CONST["num_fc_layers"]
	if args.regulariser or args.regulariser == 0.0:
		TR_CONST["regulariser"] = [(TR_CONST["regulariser"][0][0], args.regulariser)]*TR_CONST["num_layers"]
	if args.regulariser_fc or args.regulariser == 0.0:
		TR_CONST["regulariser_fc_layers"] = [(TR_CONST["regulariser_fc_layers"][0][0], args.regulariser_fc)]*TR_CONST["num_fc_layers"]
	if args.batch_normalization:
		TR_CONST["BN"] = str2bool(args.batch_normalization)
	if args.batch_normalization_fc:
		TR_CONST["BN_fc_layers"] = str2bool(args.batch_normalization_fc)
	if args.learning_rate:
		TR_CONST["learning_rate"] = args.learning_rate
	if args.debug:
		TR_CONST["debug"] = str2bool(args.debug)
	if args.output_layer:
		TR_CONST["output_activation"] = args.output_layer


	#l1, 5e3 --> stopped at 0.72 
	# TR_CONST["num_epoch"] = 2
	# for BN in [False, True]:
	# 	for BN_fc in [False, True]:
	# 		print ' *** Go with BN: %s, BN_fc: %s  ***' % (str(BN), str(BN_fc))
	# 		TR_CONST["BN"] = BN
	# 		TR_CONST["BN_fc_layers"] = BN_fc
	#  		update_setting_dict(TR_CONST)
	#  		run_with_setting(TR_CONST, sys.argv)
	
	# TR_CONST["BN"] = True
	# TR_CONST["BN_fc_layers"] = True

	# prelu, elu > lrelu > relu

	#------------------
	# TR_CONST["learning_rate"] = 1e-7
	# TR_CONST["BN"] = False
	# TR_CONST["BN_fc_layers"] = False
	# run_with_setting(TR_CONST, sys.argv)
	# TR_CONST["BN"] = True
	# TR_CONST["BN_fc_layers"] = True
	
	# TR_CONST["learning_rate"] = 3e-7

	#------------------
	update_setting_dict(TR_CONST)
	run_with_setting(TR_CONST, sys.argv)
	sys.exit()
	#------------------
	min_losses = []
	nus = [(1,4096), (1,2048), (1,256), (1,512), (1,1024), (2,64), (2,256), (3, 32)]
	for num_fc_lyr, nu in nus:
		TR_CONST["num_fc_layers"] = num_fc_lyr
		TR_CONST["nums_units_fc_layers"] = [nu]*num_fc_lyr
		min_losses.append(run_with_setting(TR_CONST, sys.argv))

	best_layer = nus[np.argmin(min_losses)]
	print 'best layer setting: ' + best_layer
	TR_CONST["num_fc_layers"] = best_layer[0]
	TR_CONST["nums_units_fc_layers"] = [best_layer[1]]*best_layer[0]
	#------------------
	min_losses = []
	num_layers = [4, 5, 6, 7, 8]
	for lyr in num_layers:
		TR_CONST["num_layers"] = lyr
		update_setting_dict(TR_CONST)
		min_losses.append(run_with_setting(TR_CONST, sys.argv))

	best_layers = num_layers[np.argmin(min_losses)]
	print 'best conv layers number: %s' % best_layers
	#------------------
	min_losses = []
	opts = ['adagrad', 'adadelta', 'adam', 'rmsprop', 'sgd']
	for opt in opts:
		if opt == 'rmsprop':
			TR_CONST["num_epoch"] = 8
		elif opt == 'sgd':
			TR_CONST["num_epoch"] = 20
		else:
			TR_CONST["num_epoch"] = 4
		TR_CONST["optimiser"] = opt
		update_setting_dict(TR_CONST)
		min_losses.append(run_with_setting(TR_CONST, sys.argv))

	best_optimiser = opts[np.argmin(min_losses)]
	print 'best optimiser: %s' % best_optimiser
	TR_CONST["optimiser"] = best_optimiser
	if best_optimiser == 'rmsprop':
		TR_CONST["num_epoch"] = 8
	elif opt == 'sgd':
		TR_CONST["num_epoch"] = 20		



