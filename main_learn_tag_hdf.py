""" To predict tags! using ilm10k data, stft or cqt representation, 
same as main_learn_tag but it load data from hdf file, 28 Dec 2015
"""
#import matplotlib
#matplotlib.use('Agg')
import argparse
import time
import sys
import os
import pdb
import numpy as np
import keras
import hyperparams_manager
from keras.utils.visualize_util import plot as keras_plot
import cPickle as cP

from constants import *
from environments import *
from training_settings import *
import my_utils
import my_plots
import my_keras_models
import my_keras_utils

def evaluate_result(y_true, y_pred):
	ret = {}
	ret['auc'] = metrics.roc_auc_score(y_true, y_pred, average='macro')
	ret['mse'] = metrics.mean_squared_error(y_true, y_pred)

	print '.'*60
	for key in ret:
		print key, ret[key]
	print '.'*60
	return ret


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
	if os.path.exists('stop_asap.keunwoo'):
		os.remove('stop_asap.keunwoo')
	
	if hyperparams["is_test"]:
		print '==== This is a test, to quickly check the code. ===='
		print 'excuted by $ ' + ' '.join(argv)
	
	auc_history = []
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
	print label_matrix.shape

	# load dataset
	train_x, valid_x, test_x, = my_utils.load_all_sets_from_hdf(tf_type=hyperparams["tf_type"],
																				n_dim=dim_latent_feature,
																				task_cla=hyperparams['isClass'])
	# *_y is not correct - 01 Jan 2016. Use numpy files directly.
	train_y, valid_y, test_y = my_utils.load_all_labels(n_dim=dim_latent_feature, 
														num_fold=10, 
														clips_per_song=3)
	threshold_label = 1.0
	if hyperparams['isClass']:
		train_y = (train_y>=threshold_label).astype(int)
		valid_y = (valid_y>=threshold_label).astype(int)
		test_y = (test_y>=threshold_label).astype(int)
	
	# print 'temporary came back with numpy loading'
	# if hyperparams["debug"]:
	# 	num_train_songs = 30
	# else:
	# 	num_train_songs = 1000
	# train_x, train_y, valid_x, valid_y, test_x, test_y = my_utils.load_all_sets(label_matrix, 
	# 																			hyperparams=hyperparams)

	hyperparams["height_image"] = train_x.shape[2]
	hyperparams["width_image"]  = train_x.shape[3]
	if hyperparams["debug"]:
		pdb.set_trace()
	
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	# train_x : (num_samples, num_channel, height, width)	
	hp_manager = hyperparams_manager.Hyperparams_Manager()
	nickname = hp_manager.get_name(hyperparams)
	timename = time.strftime('%m-%d-%Hh%M')
	if hyperparams["is_test"]:
		model_name = 'test_' + nickname
	else:
		model_name = timename + '_' + nickname
	hp_manager.save_new_setting(hyperparams)
	print '-'*60
	print 'model name: %s' % model_name
	model_name_dir = model_name + '/'
	model_weight_name_dir = 'w_' + model_name + '/'
	fileout = model_name + '_results'
	 	
	model = my_keras_models.build_convnet_model(setting_dict=hyperparams)
	model.summary()
	if not os.path.exists(PATH_RESULTS + model_name_dir):
		os.mkdir(PATH_RESULTS + model_name_dir)
		os.mkdir(PATH_RESULTS + model_name_dir + 'images/')
		os.mkdir(PATH_RESULTS + model_name_dir + 'plots/')
		os.mkdir(PATH_RESULTS_W + model_weight_name_dir)
	
	hp_manager.write_setting_as_texts(PATH_RESULTS + model_name_dir, hyperparams)
 	hp_manager.print_setting(hyperparams)

 	keras_plot(model, to_file=PATH_RESULTS + model_name_dir + 'images/'+'graph_of_model_'+hyperparams["!memo"]+'.png')
	#prepare callbacks
	weight_image_monitor = my_keras_utils.Weight_Image_Saver(PATH_RESULTS + model_name_dir + 'images/')
	patience = 3
	if hyperparams["is_test"] is True:
		patience = 99999999
	if hyperparams["isRegre"]:
		value_to_monitor = 'val_loss'
	else:
		value_to_monitor = 'val_acc'
		#history = my_keras_utils.History_Regression_Val()
	# early_stopping = keras.callbacks.EarlyStopping(monitor=value_to_monitor, 
	# 												patience=patience, 
	# 												verbose=0)
	
	# other constants
	batch_size = 32
	# if hyperparams['model_type'] == 'vgg_original':
	# 	batch_size = (batch_size * 3)/5

	predicted = model.predict(test_x, batch_size=batch_size)
	if hyperparams['debug'] == True:
		pdb.set_trace()
	print 'mean of target value:'
	if hyperparams['isRegre']:
		print np.mean(test_y, axis=0)
	else:
		print np.sum(test_y, axis=0)
	print 'mean of predicted value:'
	if hyperparams['isRegre']:
		print np.mean(predicted, axis=0)
	else:
		print np.sum(predicted, axis=0)
	print 'mse with just predicting average is %f' % np.mean((test_y - np.mean(test_y, axis=0))**2)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_init.npy', [predicted[:len(test_y)], test_y[:len(test_y)]])
	#train!	
	print '--- train starts. Remove will_stop.keunwoo to continue learning after %d epochs ---' % hyperparams["num_epoch"]
	f = open('will_stop.keunwoo', 'w')
	f.close()
	total_history = {}
	num_epoch = hyperparams["num_epoch"]
	total_epoch = 0
	
	callbacks = [weight_image_monitor]
	best_auc = 0.5

	while True:
		# [run]
		if os.path.exists('stop_asap.keunwoo'):
			print ' stop by stop_asap.keunwoo file'
			break
		history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), 
											batch_size=batch_size, 
											nb_epoch=1, 
											show_accuracy=hyperparams['isClass'], 
											verbose=1, 
											callbacks=callbacks,
											shuffle='batch')
		my_utils.append_history(total_history, history.history)
		# [validation]
		val_result = evaluate_result(valid_y, predicted) # auc
		if val_result > best_auc:
			model.save_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5")
		auc_history.append(val_result)

		print '%d-th of %d epoch is complete' % (total_epoch, num_epoch)
		total_epoch += 1
		
		# if os.path.exists('will_stop.keunwoo'):
		loss_testset = model.evaluate(test_x, test_y, show_accuracy=False, batch_size=batch_size)
		# else:
			
		# 	print ' *** will go for another one epoch. '
		# 	print ' *** $ touch will_stop.keunwoo to stop at the end of this, otherwise it will be endless.'
	#
	best_batch = np.argmax(auc_history)+1

	model.load_weights(PATH_RESULTS_W + model_weight_name_dir + "weights_best.hdf5") 

	predicted = model.predict(test_x, batch_size=batch_size)
	print 'predicted example using best model'
	print predicted[:10]
	print 'and truths'
	print test_y[:10]
	#save results
	np.save(PATH_RESULTS + model_name_dir + fileout + '_history.npy', [total_history['loss'], total_history['val_loss']])
	np.save(PATH_RESULTS + model_name_dir + fileout + '_loss_testset.npy', loss_testset)
	np.save(PATH_RESULTS + model_name_dir + 'predicted_and_truths_result.npy', [predicted, test_y])
	# np.save(PATH_RESULTS + model_name_dir + 'weights_changes.npy', np.array(weight_image_monitor.weights_changes))

	# ADD weight change saving code
	my_plots.export_history(total_history['loss'], total_history['val_loss'],
											acc=None, 
											val_acc=None, 
											out_filename=PATH_RESULTS + model_name_dir + 'plots/' + 'plots.png')
	
	
	min_loss = np.min(total_history[value_to_monitor])
	best_batch = np.argmin(total_history[value_to_monitor])+1
	num_run_epoch = len(total_history[value_to_monitor])
	oneline_result = '%s, %6.4f, %d_of_%d, %s' % (value_to_monitor, min_loss, best_batch, num_run_epoch, model_name)
	with open(PATH_RESULTS + model_name_dir + oneline_result, 'w') as f:
		pass
	f = open( (PATH_RESULTS + '%s_%s_%s_%06.4f_at_(%d_of_%d)_%s'  % \
		(timename, hyperparams["loss_function"], value_to_monitor, min_loss, best_batch, num_run_epoch, nickname)), 'w')
	f.close()
	with open('one_line_log.txt', 'a') as f:
		f.write(oneline_result)
		f.write(' ' + ' '.join(argv) + '\n')
	print '========== DONE: %s ==========' % model_name
	return min_loss

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='parser for input arguments')
	parser.add_argument('-ne', '--n_epoch', type=int, 
											help='set the number of epoch, \ndefault=30', 
											required=False)
	parser.add_argument('-tf', '--tf', help='whether cqt, stft, mfcc, melgram \ndefault=cqt.', 
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
	# do it like an approximated classification.
	TR_CONST['isClass'] = True
	TR_CONST['isRegre'] = False
	# TR_CONST['loss_function'] = 'categorical_crossentropy'
	# TR_CONST["output_activation"] = 'sigmoid'
	TR_CONST["activations"] = ['lrelu'] # alpha is 0.3 now
	TR_CONST["activations_fc_layers"] = ['lrelu']
	TR_CONST["BN"] = True
	TR_CONST["BN_fc_layers"] = True
	
	TR_CONST["!memo"] = 'batch size is 1, it is a stochastic gradient descent.'
	TR_CONST["dropouts_fc_layers"] = [0.5]
	TR_CONST["nums_units_fc_layers"] = [1024] # with 0.25 this is equivalent to 512 units
	TR_CONST["num_layers"] = 4
	TR_CONST["model_type"] = 'vgg_simple'
	TR_CONST["tf_type"] = 'melgram'

	# TR_CONST["num_fc_layers"] = 2 

	TR_CONST["BN_fc_layers"] = True
	TR_CONST["dropouts_fc_layers"] = [0.5]*max(TR_CONST["num_fc_layers"], 1)

	TR_CONST["nums_units_fc_layers"] = [4096]*max(TR_CONST["num_fc_layers"], 1)
	TR_CONST["activations_fc_layers"] = ['elu']*max(TR_CONST["num_fc_layers"], 1)
	TR_CONST["regulariser_fc_layers"] = [('l1', 0.0)] *max(TR_CONST["num_fc_layers"], 1)
	TR_CONST["act_regulariser_fc_layers"] = [('activity_l1l2', 0.0)] *max(TR_CONST["num_fc_layers"], 1)
	TR_CONST["BN_fc_layers"] = True
	TR_CONST["maxout"] = True
	TR_CONST['nb_maxout_feature'] = 4

	TR_CONST['num_sparse_layer'] = 3
	TR_CONST['maxout_sparse_layer'] = True
	TR_CONST['num_sparse_units'] = 128


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



