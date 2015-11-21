from constants import *
from environments import *
import numpy as np
import librosa
import keras
import os
import pdb

def build_convnet_model(height, width, num_labels):
	""" It builds a convnet model using keras and returns it.
	input: height: height of input image (=len_frequency)
	       width:  width of input image (=len_frame)
	"""
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation, Flatten
	from keras.layers.convolutional import Convolution2D, MaxPooling2D
	from keras.optimizers import RMSprop, SGD
	from keras.layers.normalization import LRN2D

	final_height = height
	final_width  = width

	model = Sequential()

	num_layers = 6
	image_patch_sizes = [[3,3]]*num_layers
	pool_sizes = [(2,2)]*num_layers
	num_stacks = [48]*num_layers
	dropouts = [0] + [0.25]*(num_layers-1)

	for i in xrange(num_layers):
		if i == 0:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], border_mode='same', input_shape=(2, height, width) ))
		else:
			model.add(Convolution2D(num_stacks[i], image_patch_sizes[i][0], image_patch_sizes[i][1], border_mode='same' ))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=pool_sizes[i], ignore_border=True))
		final_height = final_height / pool_sizes[i][0]
		final_width  = final_width  / pool_sizes[i][1]
		if dropouts[i] != 0:
			model.add(Dropout(dropouts[i]))
		if i != 0:
			model.add(LRN2D())

	model.add(Flatten())
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, init='normal', activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_labels, init='normal', activation='softmax'))
	rmsprop = RMSprop(lr=1e-6, rho=0.9, epsilon=1e-6)
	print '--- ready to compile keras model ---'
	model.compile(loss='mean_squared_error', optimizer=rmsprop)
	print '--- complie fin. ---'
	return model

