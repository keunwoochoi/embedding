# from constants import *
from environments import *
import numpy as np
import keras
import os
import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm, nonneg
import time

import keras.regularizers

leakage = 0.03 

#------------- Element functions ------------- #


#-------------

def build_convnet_model(setting_dict):
	start = time.time()
	loss_function = setting_dict["loss_function"]
	optimizer_name = setting_dict["optimiser"].lower() # 'SGD', 'RMSProp', ..
	learning_rate = setting_dict['learning_rate']
	#------------------------------------------------------------------#
	model_type = setting_dict["model_type"]
	if model_type.startswith('vgg'):
		model = design_2d_convnet_model(setting_dict)
	elif model_type.startswith('gnu'):
		if model_type == 'gnu_1d':
			model = design_gnu_convnet_model(setting_dict)
		elif model_type == 'gnu_mfcc':
			model = design_mfcc_convnet_model(setting_dict)
	elif model_type == 'residual':
		model = design_residual_model(setting_dict)
	#------------------------------------------------------------------#
	if optimizer_name == 'sgd':
		optimiser = SGD(lr=learning_rate, momentum=0.9, decay=1e-5, nesterov=True)
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
	model.compile(loss=loss_function, optimizer=optimiser, class_mode='binary') # mean_absolute_error, mean_squared_error, ... want to try mae later!
	until = time.time()
 	print "--- keras model was built, took %d seconds ---" % (until-start)
 	
	return model

def design_2d_convnet_model(setting_dict):

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
	# mp_strides = [(2,2)]*setting_dict['num_layers']
	#------------------------------------------------------------------#
	num_channels=1
	sigma = setting_dict['gn_sigma']	
	#-----------------------

	if model_type.startswith('vgg'):
		# layers = 4,5,6
		if model_type.startswith('vgg_modi_'): # vgg_modi_1x1, vgg_modi_3x3
			if setting_dict['tf_type'] in ['cqt', 'stft', 'melgram']:
				image_patch_sizes = [[3,3]]*num_layers
				pool_sizes = [(2,2)]*num_layers
				if num_layers == 3:
					vgg_modi_weight = [[4, 2], [8, 4], [12, 8]] 
					pool_sizes[0] = (2,5)
					pool_sizes[1] = (4,5)
					pool_sizes[2] = (4,3) # --> output: (4x2)

				elif num_layers == 4: # so that height(128) becomes 2 
					vgg_modi_weight = [[2,1], [4,2], [6,4], [8,6]]  # similar to red_pig. 'rich' setting --> later!
					pool_sizes[0] = (2,2)
					pool_sizes[1] = (2,2)
					pool_sizes[2] = (4,4)
					pool_sizes[3] = (4,4) # --> output: 2x4 melgram
					# mp_strides[0] = (2,3)
					# mp_strides[1] = (2,3)
					# mp_strides[2] = (2,3)
					# mp_strides[3] = (3,3)
					
				elif num_layers == 5:
					vgg_modi_weight = [[2,1], [4,2], [6, 4], [8, 6], [12,8]] # final layer: 8x32=256 featue maps, 
					pool_sizes[0] = (2,2) # mel input: 128x252
					pool_sizes[1] = (2,2)
					pool_sizes[2] = (2,2)
					pool_sizes[3] = (2,4)
					pool_sizes[4] = (4,4) # --> 2x2
					# mp_strides[0] = (1,1)
					# mp_strides[1] = (1,1)
					# mp_strides[2] = (1,1)
					# mp_strides[3] = (1,2) #
				elif num_layers == 6:
					vgg_modi_weight = [[2,1], [4,2], [6, 4], [8, 6], [12,8], [16,12]] # final layer: 8x32=256 featue maps, 
					pool_sizes[0] = (2,2) # mel input: 128x252
					pool_sizes[1] = (2,2)
					pool_sizes[2] = (2,2)
					pool_sizes[3] = (2,2)
					pool_sizes[4] = (2,4) 
					pool_sizes[5] = (4,4) # --> 1x1
			
		else:
			if setting_dict['tf_type'] in ['cqt', 'stft', 'melgram']:
				image_patch_sizes = [[3,3]]*num_layers
				pool_sizes = [(2,2)]*num_layers

			elif setting_dict['tf_type'] == 'mfcc':
				image_patch_sizes = [[height,1]]*num_layers
				pool_sizes = [(1,2)]*num_layers
	elif model_type.startswith('flow'):
		pass # less layers, bigger filter.
	#-------------------------------#
	# prepre modules
	model = Sequential()
	# zero-adding
	model.add(keras.layers.convolutional.ZeroPadding2D(padding=(0,2), dim_ordering='th', input_shape=(num_channels, height, width)))

	# additive gaussian noise
	if setting_dict['gaussian_noise']:
		print ' ---->>add gaussian noise '
		model.add(keras.layers.noise.GaussianNoise(sigma))

	#[Convolutional Layers]
	for conv_idx in xrange(num_layers):
		# prepare regularizer if needed.
		if setting_dict['regulariser'][conv_idx] in [None, 0.0]:
			W_regularizer = None
		else:
			if setting_dict['regulariser'][conv_idx][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser'][conv_idx][1])
				print ' ---->>prepare l2 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][conv_idx][1], conv_idx)
			elif setting_dict['regulariser'][conv_idx][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser'][conv_idx][1])
				print ' ---->>prepare l1 regulariser of %f for %d-th conv layer' % (setting_dict['regulariser'][conv_idx][1], conv_idx)

		# add conv layer
		if model_type.startswith('vgg_modi'):
			n_feat_here = int(num_stacks[conv_idx]*vgg_modi_weight[conv_idx][0])
		else:
			n_feat_here = num_stacks[conv_idx]

		if conv_idx == 0 and not setting_dict['gaussian_noise']:
			print ' ---->>First conv layer is being added! wigh %d' % n_feat_here
			model.add(Convolution2D(n_feat_here, image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same',  # no input shape after adding zero-padding
									W_regularizer=W_regularizer, init='he_normal'))

		else:
			print ' ---->>%d-th conv layer is being added with %d units' % (conv_idx, n_feat_here)
			model.add(Convolution2D(n_feat_here, image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same',	W_regularizer=W_regularizer,
									init='he_normal'))
		# add BN
		if setting_dict['BN']:
			print ' ---->>BN is added for conv layer'
			model.add(BatchNormalization(axis=1))

		# add activation
		print ' ---->>%s activation is added.' % activations[0]
		if activations[0] == 'relu':
			model.add(Activation('relu'))
		elif activations[0] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
		elif activations[0] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[0] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
	
		
		if not dropouts[conv_idx] == 0.0:
			model.add(Dropout(dropouts[conv_idx]))
			print ' ---->>Add dropout of %f for %d-th conv layer' % (dropouts[conv_idx], conv_idx)
	
		# [second conv layers] for vgg_original or vgg_modi
		if model_type in ['vgg_original', 'vgg_modi_1x1', 'vgg_modi_3x3']:

			if model_type.startswith('vgg_modi'):
				n_feat_here = int(num_stacks[conv_idx]*vgg_modi_weight[conv_idx][0])
			else:
				n_feat_here = num_stacks[conv_idx]

			if model_type == 'vgg_original':
				print ' ---->>  additional conv layer is added for vgg_original, %d' % (n_feat_here)
				model.add(Convolution2D(n_feat_here, image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
										border_mode='same',	W_regularizer=W_regularizer, init='he_normal'))
			elif model_type == 'vgg_modi_1x1':
				print ' ---->>  additional conv layer is added for vgg_modi_1x1, %d' % (n_feat_here)
				model.add(Convolution2D(n_feat_here, 1, 1, 
										border_mode='same',	W_regularizer=W_regularizer, init='he_normal'))
			elif model_type == 'vgg_modi_3x3':
				print ' ---->>  additional conv layer is added for vgg_modi_3x3, %d' % (n_feat_here)
				model.add(Convolution2D(n_feat_here, image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
										border_mode='same',	W_regularizer=W_regularizer, init='he_normal'))

			if setting_dict['BN']:
				print ' ---->>  and BN,'
				model.add(BatchNormalization(axis=1))
			# add activation
			
			print ' ---->>  %s activation is added.' % activations[0]
			if activations[0] == 'relu':
				model.add(Activation('relu'))
			elif activations[0] == 'lrelu':
				model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
			elif activations[0] == 'prelu':
				model.add(keras.layers.advanced_activations.PReLU())
			elif activations[0] == 'elu':
				model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		
		#[third conv layer] for vgg_modi_3x3
		if model_type == 'vgg_modi_3x3':
			n_feat_here = int(num_stacks[conv_idx]*vgg_modi_weight[conv_idx][1])
			print ' ---->>     one more additional 1x1 conv layer is added for vgg_modi_3x3, %d' % (n_feat_here)
			model.add(Convolution2D(n_feat_here, 1, 1, 
									border_mode='same',	W_regularizer=W_regularizer, init='he_normal'))
			if setting_dict['BN']:
				print ' ---->>    and BN + elu'
				model.add(BatchNormalization(axis=1))
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))

		# add pooling
		if model_type in ['vgg_original', 'vgg_modi_1x1', 'vgg_modi_3x3']:
			print ' ---->>MP is added', pool_sizes[conv_idx]
			model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
		elif model_type.startswith('vgg_simple'):
			print ' ---->>MP is added', pool_sizes[conv_idx]
			model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
		if model_type == 'vgg_simple':
			# add dropout
			if not dropouts[conv_idx] == 0.0:
				model.add(Dropout(dropouts[conv_idx]))
				print ' ---->>Add dropout of %f for %d-th conv layer' % (dropouts[conv_idx], conv_idx)
		
	#[Fully Connected Layers]
	model.add(Flatten())
	for fc_idx in xrange(num_fc_layers):
		# setup regulariser
		if setting_dict['regulariser_fc_layers'][fc_idx] is None:
			W_regularizer = None
		else:
			if setting_dict['regulariser_fc_layers'][fc_idx][0] == 'l2':
				W_regularizer=keras.regularizers.l2(setting_dict['regulariser_fc_layers'][fc_idx][1])
			elif setting_dict['regulariser_fc_layers'][fc_idx][0] == 'l1':
				W_regularizer=keras.regularizers.l1(setting_dict['regulariser_fc_layers'][fc_idx][1])
		# maxout...
		if setting_dict['maxout']:
			nb_feature = 8
			model.add(MaxoutDense(nums_units_fc_layers[fc_idx], nb_feature=nb_feature ,W_regularizer=W_regularizer))
			print ' --->>MaxoutDense added with %d output units, %d features' % (nums_units_fc_layers[fc_idx], nb_feature)
		else:
			
			# ..or, dense layer
			if not dropouts_fc_layers[fc_idx] == 0.0:
				print ' ---->>Dense layer, %d, is added with dropout of %f.' % (nums_units_fc_layers[fc_idx], dropouts_fc_layers[fc_idx])
				model.add(Dense(nums_units_fc_layers[fc_idx],init='he_normal'))
			
			else:
				print ' ---->>Dense layer, %d, is added with regularizer.' % nums_units_fc_layers[fc_idx]
				model.add(Dense(nums_units_fc_layers[fc_idx], W_regularizer=W_regularizer,
														init='he_normal'))
			
			# Activations
			print ' ---->>%s activation is added.' % activations[0]
			if activations[0] == 'relu':
				model.add(Activation('relu'))
			elif activations[0] == 'lrelu':
				model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
			elif activations[0] == 'prelu':
				model.add(keras.layers.advanced_activations.PReLU())
			elif activations[0] == 'elu':
				model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		
		# Dropout
		if not dropouts_fc_layers[fc_idx] == 0.0:
			model.add(Dropout(dropouts_fc_layers[fc_idx]))
			print ' ---->>Dropout for fc layer, %f' % dropouts_fc_layers[fc_idx]
		# BN
		if setting_dict['BN_fc_layers']:
			print ' ---->>BN for dense is added'
			model.add(BatchNormalization()) ## BN vs Dropout - who's first?
		
	#[Output layer]
	if setting_dict["output_activation"]:
		print ' ---->>Output dense and activation is: %s with %d units' % (setting_dict["output_activation"], num_labels)
		model.add(Dense(num_labels, activation=setting_dict["output_activation"],
									init='he_normal')) 
	else:
		print ' ---->>Output dense and activation: linear with %d units' % num_labels
		model.add(Dense(num_labels, activation='linear')) 

	return model	


def design_residual_model(setting_dict):
	'''residual net using graph'''
	pass



def design_gnu_convnet_model(setting_dict):
	'''It's a hybrid type model - perhaps something like Sander proposed?
	Mainly convnet is done as 1d time-axis, then 
	'''
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
	#------------------------------------------------------------------#
	num_channels=1

	image_patch_sizes = [[1,3], [1,4], [1,4],[1,4]]
	pool_sizes = [(1,3), (1,4), (1,4),(2,4)]
	num_stacks = [48,48,64,64]

	model = Sequential()

	for conv_idx in range(len(num_stacks)):
		if conv_idx == 0:
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
										border_mode='same', 
										input_shape=(1, height, width), 
										init='he_normal'))
		else:
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
										border_mode='same', 
										init='he_normal'))

		model.add(BatchNormalization())
		model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
		model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))

	model.add(Flatten())

	model.add(Dense(2048, init='he_normal'))
	model.add(Dropout(0.5))
	model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
	model.add(BatchNormalization())

	model.add(Dense(2048, init='he_normal'))
	model.add(Dropout(0.5))
	model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
	model.add(BatchNormalization())

	model.add(Dense(num_labels, activation='sigmoid',
								init='he_normal')) 
	return model
		

def design_mfcc_convnet_model(setting_dict):
	height = setting_dict["height_image"]
	width = setting_dict["width_image"]
	num_labels = setting_dict["dim_labels"]
	#------------------------------------------------------------------#
	num_channels=1
	image_patch_sizes = [[height/3,1], [1,1], [1,1], [1,1]]
	pool_sizes = [(1,3), (1,4), (1,4), (1,4)]
	num_stacks = [48, 48, 64, 64]

	model = Sequential()

	for conv_idx in range(len(num_stacks)):
		if conv_idx == 0:
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='valid', 
									input_shape=(1, height, width), 
									subsample=(height/3, 1),
									init='he_normal'))
		else:
			model.add(Convolution2D(num_stacks[conv_idx], image_patch_sizes[conv_idx][0], image_patch_sizes[conv_idx][1], 
									border_mode='same', 
									init='he_normal'))
		model.add(BatchNormalization())
		model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
		model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx]))
	
	model.add(Flatten())

	model.add(Dense(2048, init='he_normal'))
	model.add(Dropout(0.5))
	model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
	model.add(BatchNormalization())

	model.add(Dense(2048, init='he_normal'))
	model.add(Dropout(0.5))
	model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
	model.add(BatchNormalization())

	model.add(Dense(num_labels, activation='sigmoid',
								init='he_normal')) 
	return model
	




#--------------------------------------------#

def design_1d_time_convnet_model(setting_dict):
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
	#------------------------------------------------------------------#
	num_channels=1

	if setting_dict['tf_type'] in ['cqt', 'stft', 'melgram']:
		image_patch_sizes = [1,5]*2 + [1,3]*(num_layers-1)
		pool_sizes = [(2,2)]*num_layers
	elif setting_dict['tf_type'] == 'mfcc':
		image_patch_sizes = [[height,1]]*num_layers
		pool_sizes = [(1,2)]*num_layers	

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
			elif setting_dict['regulariser'][conv_idx][0] == 'l1':
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
		# add BN
		if setting_dict['BN']:
			print ' ---->>BN is added for conv layer'
			model.add(BatchNormalization())

		# add activation
		print ' ---->>%s activation is added.' % activations[conv_idx]
		if activations[conv_idx] == 'relu':
			model.add(Activation('relu'))
		elif activations[conv_idx] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
		elif activations[conv_idx] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations[conv_idx] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print ' ---->>No activation here? No!'

		# add pooling	
		print ' ---->>MP with (2,2) strides is added', pool_sizes[conv_idx]
		model.add(MaxPooling2D(pool_size=pool_sizes[conv_idx], strides=(2, 2)))
		# add dropout
		if not dropouts[conv_idx] == 0.0:
			model.add(Dropout(dropouts[conv_idx]))
			print ' ---->>Add dropout of %f for %d-th conv layer' % (dropouts[conv_idx], conv_idx)
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
		
		else:
			print ' ---->>Dense layer is added with regularizer.'
			model.add(Dense(nums_units_fc_layers[fc_idx], W_regularizer=W_regularizer,
													init='he_normal'))
		
		# Activations
		print ' ---->>%s activation is added' % activations_fc_layers[fc_idx]
		if activations_fc_layers[fc_idx] == 'relu':
			model.add(Activation('relu'))
		elif activations_fc_layers[fc_idx] == 'lrelu':
			model.add(keras.layers.advanced_activations.LeakyReLU(alpha=leakage))
		elif activations_fc_layers[fc_idx] == 'prelu':
			model.add(keras.layers.advanced_activations.PReLU())
		elif activations_fc_layers[fc_idx] == 'elu':
			model.add(keras.layers.advanced_activations.ELU(alpha=1.0))
		else:
			print ' ---->>No activation here? No!'
		# Dropout
		if not dropouts_fc_layers[fc_idx] == 0.0:
			model.add(Dropout(dropouts_fc_layers[fc_idx]))
		# BN
		if setting_dict['BN_fc_layers']:
			print ' ---->>BN for dense is added'
			model.add(BatchNormalization())

	#[Output layer]
	if setting_dict["output_activation"]:
		print ' ---->>Output dense and activation is: %s with %d units' % (setting_dict["output_activation"], num_labels)
		model.add(Dense(num_labels, activation=setting_dict["output_activation"],
									init='he_normal')) 
	else:
		print ' ---->>Output dense and activation: linear with %d units' % num_labels
		model.add(Dense(num_labels, activation='linear')) 

	return model

