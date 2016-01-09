import keras.callbacks
import my_plots
from constants import *
from environments import *
import pdb
import time
import numpy as np
from keras.utils import np_utils

'''
class Keras_Results():
	"""It is a class to contain every information about a learning result.
	Goal: to make it easier to see the result, compare results, resume from previous model, ...

	"""
	def __init__(self):
		self.model_name = None
		self.adjspecies = None
		self.result_path = None # a 'mother' path that contains all - images, results, model, ...
		self.losses
		self.accs
		self.val_losses
		self.val_accs
		self.test_losses
		self.test_accs
		self.test_predicted
		self.test_truths
		self.settings = {}
	
	def compare(self, another_result):

	def show_plots(self):
		"""show plots - loss, etc."""
	
	def load_model(self):
		"""return the corresponding keras model"""
'''

class History_Classification(keras.callbacks.Callback):
	"""history, not validation. use History_Val to include both training and validation data"""
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accs = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accs.append(logs.get('acc'))

class History_Classification_Val(keras.callbacks.Callback):
	"""history with valudation data"""
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accs = []
		self.val_losses = []
		self.val_accs = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accs.append(logs.get('acc'))
		self.val_losses.append(logs.get('val_loss'))
		self.val_accs.append(logs.get('val_accuracy'))

class History_Regression(keras.callbacks.Callback):
	"""history, not validation. use History_Val to include both training and validation data"""
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

class History_Regression_Val(keras.callbacks.Callback):
	"""history with valudation data"""
	def on_train_begin(self, logs={}):
		self.losses = []
		self.val_losses = []

	def on_epoch_end(self, batch, logs={}):
		pdb.set_trace()
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

class Weight_Image_Saver(keras.callbacks.Callback):
	def __init__(self, path_to_save):
		self.path_to_save = path_to_save
		self.recent_weights = None
		self.weights_changes = []

	def on_train_begin(self, logs={}):
		pass
		# seconds = str(int(time.time()))
		# my_plots.save_model_as_image(self.model, save_path=self.path_to_save, 
		# 										filename_prefix='000_INIT_', 
		# 										normalize='local', 
		# 										mono=True)
		# my_plots.save_model_as_image(self.model, save_path=self.path_to_save, 
		# 										filename_prefix='INIT_', 
		# 										normalize='local', 
		# 										mono=True

	def on_train_end(self, logs={}):
		my_plots.save_weights_changes_plot(self.weights_changes, self.path_to_save)

	def on_epoch_begin(self, epoch, logs={}):
		# load weight into self.recent_weight
		self.recent_weights = self.load_weights()

	def on_epoch_end(self, epoch, logs={}):
		#seconds = str(int(time.time()))
		my_plots.save_model_as_image(self.model, save_path=self.path_to_save, 
												filename_prefix='', 
												normalize='local', 
												mono=True)
		average_change_per_layer = self.get_weights_change()
		print 'average change per layer:'
		print average_change_per_layer
		self.weights_changes.append(average_change_per_layer)

	def load_weights(self):
		ret_W = []
		for layerind, layer in enumerate(self.model.layers):
			g = layer.get_config()
			if g['name'] in ['Convolution2D', 'Convolution1D', 'Dense']:
				# W = layer.get_weights()[0] # tensor. same as layer.W.get_value(borrow=True)
				ret_W.append(layer.W.get_value(borrow=True))
				#W = np.squeeze(W)
		return ret_W

	def get_weights_change(self):
		# compute average amount of change by comparing current weight and self.recent_weight
		current_weights = self.load_weights()
		num_layer = len(current_weights)
		ret = [0.0] * num_layer
		for layer_idx in num_layer:
			ret[layer_idx] = np.mean(np.divide(np.abs(self.recent_weights[layer_idx] - current_weights[layer_idx]),  np.abs(self.recent_weights[layer_idx])))
		return ret


def continuous_to_categorical(y):
	'''input y: continuous label, (N,M) array.
	return: (N,M) array.
	'''
	maxind = np.argmax(y, axis=1)
	return np_utils.to_categorical(maxind, y.shape[1])


