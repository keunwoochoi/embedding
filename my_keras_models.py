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

def build_regression_convnet_model(setting_dict):
	
	is_test = setting_dict["is_test"]
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
	# prepre modules
	model = Sequential()
	#[Convolutional Layers]
	for conv_idx in xrange(num_layers):
		if setting_dict['regulariser'][conv_idx] in [None, 0.0]:
			W_regularizer = None
		else:
			if setting_dict['regulariser'][conv_idx][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser'][conv_idx][1])
				print ' ---->>prepare l2 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][conv_idx][1], conv_idx)
			elif setting_dict['regulariser'][i][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser'][conv_idx][1])
				print ' ---->>prepare l1 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][conv_idx][1], conv_idx)

		# add conv layer
		if conv_idx == 0:
			print ' ---->>First conv layer is being added!'
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same', 
									input_shape=(num_channels, height, width), 
									W_regularizer=W_regularizer,
									init='he_normal'))

		else:
			print ' ---->>%d-th conv layer is being added ' % conv_idx
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same',
									W_regularizer=W_regularizer,
									init='he_normal'))

		if setting_dict['BN']:
			print ' ---->>BN is added for conv layer'
			model.add(BatchNormalization())
		# add activation
		print ' ---->>%s activation is added.' % activations[conv_idx]
		if activations[conv_idx] == 'relu':
			model.add(Activation('relu'))
		elif activations[conv_idx] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
		elif activations[conv_idx] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[conv_idx] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print ' ---->>No activation here? No!'
		# add dropout
		if not dropouts[conv_idx] == 0.0:
			model.add(Dropout(dropouts[conv_idx]))
			print ' ---->>Add dropout of %f for %d-th conv layer' % (dropouts[conv_idx], conv_idx)
		
		if model_type.startswith('vgg_original'):
			print ' ---->>additional conv layer is added for vgg_original'
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same',
									W_regularizer=W_regularizer,
									init='he_normal'))
			if setting_dict['BN']:
				print ' ---->>and BN,'
				model.add(BatchNormalization())
			# add activation
			print ' ---->>and %s activaion.' % activations[conv_idx]
			if activations[conv_idx] == 'relu':
				model.add(Activation('relu'))
			elif activations[conv_idx] == 'lrelu':
				model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
			elif activations[conv_idx] == 'prelu':
				model.add(keras.layers.advanced_activations.PReLU())
			elif activations[conv_idx] == 'elu':
				model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
			else:
				print ' ---->>No activation here? No!'
			# add dropout
			if not dropouts[conv_idx] == 0.0:
				model.add(Dropout(dropouts[conv_idx]))
				print ' ---->>Add dropout of %f for %d-th conv layer' % (dropouts[conv_idx], conv_idx)

		# add pooling
		if model_type.startswith('vgg_original'):
			print ' ---->>MP with (2,2) strides is added', pool_sizes[conv_idx]
			model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx], strides=(2, 2)))
		elif model_type.startswith('vgg_simple'):
			print ' ---->>MP is added', pool_sizes[conv_idx]
			model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
		
	#[Fully Connected Layers]
	model.add(Flatten())
	for fc_idx in xrange(num_fc_layers):
		if setting_dict['regulariser_fc_layers'][fc_idx] is None:
			W_regularizer = None
		else:
			if setting_dict['regulariser_fc_layers'][fc_idx][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser_fc_layers'][fc_idx][1])
			elif setting_dict['regulariser_fc_layers'][fc_idx][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser_fc_layers'][fc_idx][1])
		# dense layer
		if not dropouts_fc_layers[fc_idx] == 0.0:
			print ' ---->>Dense layer is added with dropout of %f.' % dropouts_fc_layers[fc_idx]
			model.add(Dense(nums_units_fc_layers[fc_idx],init='he_normal'))
			model.add(Dropout(dropouts_fc_layers[fc_idx]))
		else:
			print ' ---->>Dense layer is added with regularizer.'
			model.add(Dense(nums_units_fc_layers[fc_idx], W_regularizer=W_regularizer,
													init='he_normal'))
		# BN
		if setting_dict['BN_fc_layers']:
			print ' ---->>BN for dense is added'
			model.add(BatchNormalization())
		# Activations
		print ' ---->>%s activation is added' % activations[fc_idx]
		if activations[fc_idx] == 'relu':
			model.add(Activation('relu'))
		elif activations[fc_idx] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.1))
		elif activations[fc_idx] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[fc_idx] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print ' ---->>No activation here? No!'
	if setting_dict["output_activation"]:
		print ' ---->>Output dense and activation is: %s with %d units' % (setting_dict["output_activation"], num_labels)
		model.add(Dense(num_labels, activation=setting_dict["output_activation"],
									init='he_normal')) 
	else:
		print ' ---->>Output dense and activation: linear with %d units' % num_labels
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
	print ' ---->>--- ready to compile keras model ---'
	model.compile(loss=loss_function, optimizer=optimiser) # mean_absolute_error, mean_squared_error, ... want to try mae later!
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

