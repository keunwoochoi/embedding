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

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

class Weight_Image_Saver(keras.callbacks.Callback):
	def __init__(self, path_to_save):
		self.path_to_save = path_to_save
		
	def on_train_begin(self, logs={}):
		seconds = str(int(time.time()))
		my_plots.save_model_as_image(self.model, save_path=self.path_to_save, 
												filename_prefix=seconds+'_INIT_', 
												normalize='local', 
												mono=False)

	def on_epoch_end(self, batch, logs={}):
		seconds = str(int(time.time()))
		my_plots.save_model_as_image(self.model, save_path=self.path_to_save, 
												filename_prefix=seconds+'_', 
												normalize='local', 
												mono=False)

def continuous_to_categorical(y):
	'''input y: continuous label, (N,M) array.
	return: (N,M) array.
	'''
	maxind = np.argmax(y, axis=1)
	return np_utils.to_categorical(maxind, y.shape[1])


