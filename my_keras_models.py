# from constants import *
from environments import *
import numpy as np
import keras
import os
import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm, nonneg


import keras.regularizers

def build_regression_convnet_model(setting_dict, is_test):
	
	height = setting_dict["height_image"]
	width = setting_dict["width_image"]
	dropouts = setting_dict["dropouts"]
	num_labels = setting_dict["dim_labels"]
	num_layers = setting_dict["num_layers"]
	activations = setting_dict["activations"] #
	model_type = setting_dict["model_type"] # not used now.
	num_stacks = setting_dict["num_feat_maps"]

	num_fc_layers = setting_dict["num_fc_layers"]
	dropouts_fc_layers = setting_dict["dropouts_fc_layers"]
	nums_units_fc_layers = setting_dict["nums_units_fc_layers"]
	activations_fc_layers = setting_dict["activations_fc_layers"]
	
	loss_function = setting_dict["loss_function"]
	optimizer_name = setting_dict["optimiser"].lower() # 'SGD', 'RMSProp', ..
	#------------------------------------------------------------------#
	num_channels=1

	if model_type.startswith('vgg'):
		# layers = 4,5,6
		if setting_dict['tf_type'] in ['cqt', 'stft']:
			image_patch_sizes = [[3,3]]*num_layers
			pool_sizes = [(2,2)]*num_layers
			if num_layers < 5:
				pool_sizes[-1] = (3,3)
		elif setting_dict['tf_type'] == 'mfcc':
			image_patch_sizes = [[height,1]]*num_layers
			pool_sizes = [(1,2)]*num_layers
	elif model_type.startswith('flow'):
		pass # less layers, bigger filter.

	if setting_dict['tf_type'] == 'mfcc':
		learning_rate = 1e-6
	elif setting_dict['tf_type'] in ['stft', 'cqt']:
		learning_rate = 1e-6
	else:
		learning_rate = 1e-6
	#-------------------------------#
	model = Sequential()
	#[Convolutional Layers]
	for i in xrange(num_layers):
		if setting_dict['regulariser'][i] in [None, 0.0]:
			W_regularizer = None
		else:
			if setting_dict['regulariser'][i][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser'][i][1])
				print 'Add l2 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][i][1], i)
			elif setting_dict['regulariser'][i][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser'][i][1])
				print 'Add l1 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][i][1], i)

		# add conv layer
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
									border_mode='same', 
									input_shape=(num_channels, height, width), 
									W_regularizer=W_regularizer))

		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
									border_mode='same',
									W_regularizer=W_regularizer))

		# BN - according to the article on Residual Net (which follows the original paper.)
		if setting_dict['BN']:
			model.add(BatchNormalization())
		# add activation
		if activations[i] == 'relu':
			model.add(Activation('relu'))
		elif activations[i] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
		elif activations[i] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[i] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print 'No activation here? No!'
		# add dropout
		if not dropouts[i] == 0.0:
			model.add(Dropout(dropouts[i]))
			print 'Add dropout of %f for %d-th conv layer' % (dropouts[i], i)
		else:
			pass
		# add pooling
		model.add(MaxPooling2D(pool_size=pool_sizes[i]))
		
	#[Fully Connected Layers]
	model.add(Flatten())
	for j in xrange(num_fc_layers):
		if setting_dict['regulariser_fc_layers'][j] is None:
			W_regularizer = None
		else:
			if setting_dict['regulariser_fc_layers'][j][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser_fc_layers'][j][1])
			elif setting_dict['regulariser_fc_layers'][j][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser_fc_layers'][j][1])
		# dense layer
		if not dropouts_fc_layers[j] == 0.0:
			model.add(Dense(nums_units_fc_layers[j]))
			model.add(Dropout(dropouts_fc_layers[j]))
		else:
			model.add(Dense(nums_units_fc_layers[j], W_regularizer=W_regularizer))
		# BN
		if setting_dict['BN_fc_layers']:
			model.add(BatchNormalization())
		# Activations
		if activations[i] == 'relu':
			model.add(Activation('relu'))
		elif activations[i] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
		elif activations[i] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[i] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print 'No activation here? No!'
	if setting_dict["output_activation"]:
		print 'Output activation is: ' + 	setting_dict["output_activation"]
		model.add(Dense(num_labels, activation=setting_dict["output_activation"])) 
	else:
		print 'Output activation: linear'
		model.add(Dense(num_labels, activation='linear')) 

	if optimizer_name == 'sgd':
		optimiser = SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
	elif optimizer_name == 'rmsprop':
		optimiser = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
	elif optimizer_name == 'adagrad':
		optimiser = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
	elif optimizer_name == 'adadelta':
		optimiser = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
	elif optimizer_name == 'adam':
		optimiser = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	else:
		raise RuntimeError('no optimiser? no! - %s' % optimizer_name )
	# print '--- ready to compile keras model ---'
	model.compile(loss=loss_function, optimizer=optimiser) # mean_absolute_error, mean_squared_error, ... want to try mae later!
	# print '--- complie fin. ---'
	return model

def build_whole_graph():
	'''A graph model that takes advantages of
	 - CQT 
	 - chromagram
	 - harmonigram
	 - pitchgram
	 - MFCC & friends
	 - '''
 	from keras.models import Graph
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.optimizers import RMSprop, SGD
	from keras.layers.normalization import LRN2D

	graph = Graph()
	# graph.add_input(name='cqt_mono', input_shape=(blah))
	# graph.add_input(name='mfcc_mono', input_shape=(19*3, ))
	# graph_add_input(name='chroma_mono', input_shape=(,))
	# graph.add_input(name='stft_mono', input_shape=(,))

