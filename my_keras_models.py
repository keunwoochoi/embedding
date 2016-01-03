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

import keras.regularizers

def build_convnet_model(height, width, num_labels, num_layers=4):
	""" It builds a convnet model using keras and returns it.
	input: height: height of input image (=len_frequency)
	       width:  width of input image (=len_frame)
	"""
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.optimizers import RMSprop
	
	model = Sequential()

	image_patch_sizes = [[3,3]]*num_layers
	if num_layers <= 5:
		pool_sizes = [(3,3)]*(2) + [(2,2)]*(num_layers-2)
	else:
		pool_sizes = [(2,2)]*num_layers

	num_stacks = [48]*num_layers
	dropouts = [0] + [0.25]*(num_layers-1)

	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', input_shape=(2, height, width) ))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same' ))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=pool_sizes[i], border_mode='same'))
		# final_height = final_height / pool_sizes[i][0]
		# final_width  = final_width  / pool_sizes[i][1]
		if dropouts[i] != 0:
			model.add(Dropout(dropouts[i]))
		# if i != 0:
		# 	model.add(LRN2D())

	model.add(Flatten())
	model.add(Dense(1024, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Dense(num_labels, init='normal', activation='linear'))
	rmsprop = RMSprop(lr=3e-6, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='mean_squared_error', optimizer=rmsprop)
	print '--- complie fin. ---'
	return model

def build_classification_convnet_model(height, width, num_labels, num_layers=5, model_type='vgg', num_channels=1):
	''' should add BN'''
	model = Sequential()
	image_patch_sizes = [[3,3]]*num_layers
	pool_sizes = [(2,2)]*num_layers
	
	final_height, final_width = height, width

	num_stacks = [48]*1 + [48]*(num_layers-1)
	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', 
								input_shape=(num_channels, height, width), 
								activation='relu' ))
		else:
			# model.add(BatchNormalization())
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', 
								activation='relu'))
		final_height = final_height / pool_sizes[i][0]
		final_width  = final_width  / pool_sizes[i][1]

		model.add(MaxPooling2D(pool_size=pool_sizes[i]))
		model.add(Dropout(0.25))
	print 'Input height,width: %d,%d, at the end of conv layer, h,w:%d,%d' \
								% (height, width, final_height, final_width)
	print ', so the flatten layer has %d units' % (final_height*final_width*num_stacks[-1])
	model.add(Flatten())
	
	# model.add(BatchNormalization())
	model.add(Dense(1024, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	
	# model.add(BatchNormalization())
	model.add(Dense(1024, init='normal', activation='relu'))
	model.add(Dropout(0.25))

	#model.add(BatchNormalization())
	model.add(Dense(num_labels, init='normal', activation='softmax'))
	optimiser = SGD(lr=1e-5, momentum=0.9, decay=1e-6, nesterov=True)
	#rmsprop = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='categorical_crossentropy', optimizer=optimiser) # mean_absolute_error, mean_squared_error, ...
	print '--- complie fin. ---'
	return model


def build_regression_convnet_model(height, width, dropouts, num_labels, num_layers=5, model_type='vgg', num_channels=1):
	
	
	model = Sequential()
	image_patch_sizes = [[3,3]]*num_layers
	pool_sizes = [(2,2)]*num_layers

	num_stacks = [64]*1 + [64]*(num_layers-1)
	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
									border_mode='same', 
									input_shape=(num_channels, height, width), 
									activation='tanh' ))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], border_mode='same', activation='tanh'))
		model.add(MaxPooling2D(pool_size=pool_sizes[i]))
		model.add(Dropout(dropouts[i]))

	model.add(Flatten())
	model.add(Dense(1024, init='normal', activation='tanh', W_regularizer=keras.regularizers.l1(0.01)))
	#model.add(Dropout(0.25))

	model.add(Dense(1024, init='normal', activation='tanh', W_regularizer=keras.regularizers.l1(0.01)))
	# model.add(Dropout(0.25))

	model.add(Dense(num_labels, init='normal', activation='linear', W_regularizer=keras.regularizers.l1(0.01)))
	optimiser = SGD(lr=3e-5, momentum=0.9, decay=1e-6, nesterov=True)
	#rmsprop = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='mean_squared_error', optimizer=optimiser) # mean_absolute_error, mean_squared_error, ... want to try mae later!
	print '--- complie fin. ---'
	return model


def build_strict_convnet_model(height, width, num_labels, num_layers=5, model_type='vgg'):
	""" It builds a convnet model using keras and returns it.
	input: height: height of input image (=len_frequency)
	       width:  width of input image (=len_frame)

	*** this model uses relu and dropout. At 1 Dec 2015, it fails and converges to make almost-zero outputs.
	***
	*** This 'strict' model assumes the model should be sensitive to different frequency gaps.
	*** i.e. at the first (and second) layer, no pooling can be done on frequency axis.
	***
	"""
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.optimizers import RMSprop, SGD
	
	model = Sequential()
	if model_type == 'vgg':
		image_patch_sizes = [[3,3]]*num_layers
		pool_sizes = [(2,2)]*num_layers
	elif model_type == 'gnu':
		if num_layers == 5:
			image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
			pool_sizes = [(1,3)] + [(2,3)] + [(3,2)]*(num_layers-2)# 168/(1,1,3,3,3)=6, 256/(3,3,2,2,2)=3
		elif num_layers == 4:
			image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
			pool_sizes = [(1,4)]*(2) + [(4,2)]*(num_layers-2)
		elif num_layers == 6:
			image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
			pool_sizes = [(1,3)]*(2) + [(2,2)]*2 + [(3,1)]*2

	num_stacks = [64]*1 + [64]*(num_layers-1)
	dropouts = [0.25]*2 + [0.25]*(num_layers-2)

	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', input_shape=(2, height, width), activation='relu' ))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', activation='relu'))
		model.add(MaxPooling2D(pool_size=pool_sizes[i], border_mode='same'))
		# final_height = final_height / pool_sizes[i][0]
		# final_width  = final_width  / pool_sizes[i][1]
		if dropouts[i] != 0:
			model.add(Dropout(dropouts[i]))
		# if i != 0:
		# 	model.add(LRN2D())

	model.add(Flatten())
	model.add(Dense(1024, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Dense(1024, init='normal', activation='relu'))
	model.add(Dropout(0.25))
	
	model.add(Dense(num_labels, init='normal', activation='linear'))
	rmsprop = RMSprop(lr=1e-7, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='mean_absolute_error', optimizer=rmsprop) # mean_absolute_error, mean_squared_error, ...
	print '--- complie fin. ---'
	return model

def build_overfitting_convnet_model(height, width, num_labels, num_layers=5):
	""" It builds a convnet model using keras and returns it.
	input: height: height of input image (=len_frequency)
	       width:  width of input image (=len_frame)

	***
	*** This 'strict' model assumes the model should be sensitive to different frequency gaps.
	*** i.e. at the first (and second) layer, no pooling can be done on frequency axis.
	***
	"""
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.optimizers import RMSprop, SGD
	print "==== INTERNTIONALLY OVERFITTING MODEL! ===="
	model = Sequential()
	if num_layers == 5:
		image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
		pool_sizes = [(1,3)] + [(2,3)] + [(3,2)]*(num_layers-2)# 168/(1,1,3,3,3)=6, 256/(3,3,2,2,2)=3
	elif num_layers == 4:
		image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
		pool_sizes = [(1,4)]*(2) + [(4,2)]*(num_layers-2)
	elif num_layers == 6:
		image_patch_sizes = [[10,3]] + [[10,3]] + [[3,3]]*(num_layers-2) 
		pool_sizes = [(1,3)]*(2) + [(2,2)]*2 + [(3,1)]*2

	num_stacks = [64]*num_layers
	dropouts = [0]*num_layers

	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same', input_shape=(2, height, width) ))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], 
								border_mode='same' ))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=pool_sizes[i], border_mode='same'))
		# final_height = final_height / pool_sizes[i][0]
		# final_width  = final_width  / pool_sizes[i][1]
		if dropouts[i] != 0:
			model.add(Dropout(dropouts[i]))
		# if i != 0:
		# 	model.add(LRN2D())

	model.add(Flatten())
	model.add(Dense(512, init='normal', activation='relu'))
	model.add(Dense(512, init='normal', activation='relu'))

	model.add(Dense(num_labels, init='normal', activation='linear'))
	rmsprop = RMSprop(lr=5e-5, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='mean_absolute_error', optimizer=rmsprop) # mean_absolute_error, mean_squared_error, ...
	print '--- complie fin. ---'
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
	graph.add_input(name='cqt_all_mono', input_shape=(blah))
	graph.add_input(name='cqt_har_mono', input_shape=(blah))
	graph.add_input(name='cqt_per_mono', input_shape=(blah))
	graph.add_input(name='mfcc_mono', input_shape=(19*3, ))
	graph_add_input(name='pitchgram_mono', input_shape=())
	graph_add_input(name='chroma_mono', input_shape=())









	
