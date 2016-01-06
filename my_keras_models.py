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

	model = Sequential()
		
	if setting_dict['tf_type'] in ['cqt', 'stft']:
		image_patch_sizes = [[3,3]]*num_layers
		pool_sizes = [(2,2)]*num_layers
	elif setting_dict['tf_type'] == 'mfcc':
		image_patch_sizes = [[height,1]]*num_layers
		pool_sizes = [(1,2)]*num_layers

	if setting_dict['tf_type'] == 'mfcc':
		learning_rate = 1e-7
	elif setting_dict['tf_type'] in ['stft', 'cqt']:
		learning_rate = 1e-6
	else:
		learning_rate = 1e-6
	#-------------------------------#
	
	W_regularizer=keras.regularizers.l2(0.0002)
	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
									border_mode='same', 
									input_shape=(num_channels, height, width), 
									activation=activations[i],
									W_regularizer=W_regularizer))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
									border_mode='same', 
									activation=activations[i],
									W_regularizer=W_regularizer))
		model.add(MaxPooling2D(pool_size=pool_sizes[i]))
		if not is_test:
			if not dropouts[i] == 0.0:
				model.add(Dropout(dropouts[i]))
			else:
				print ' ...no dropout but I put reguralisation.'
		else:
			print ' ...no dropout in test'

	model.add(Flatten())
	for j in xrange(num_fc_layers):
		if is_test:
			model.add(Dense(nums_units_fc_layers[j], activation=activations_fc_layers[j]))
			print ' ...and no reguralisation in test.'
		else:
			if not dropouts_fc_layers[j] == 0.0:
				model.add(Dense(nums_units_fc_layers[j], activation=activations_fc_layers[j]))
				model.add(Dropout(dropouts_fc_layers[j]))
			else:
				model.add(Dense(nums_units_fc_layers[j], activation=activations_fc_layers[j], 
														W_regularizer=keras.regularizers.l2(0.0002)))
				print ' ...no dropout but I put reguralisation.'

	model.add(Dense(num_labels, activation='linear', W_constraint = nonneg())) 

	if optimizer_name == 'sgd':
		optimiser = SGD(lr=learning_rate, momentum=0.9, decay=1e-6, nesterov=True)
	elif optimizer_name == 'rmsprop':
		optimiser = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-6)
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

