import keras.callbacks
import my_plots
from constants import *
from environments import *
import pdb

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
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

class Weight_Image_Saver(keras.callbacks.Callback):
	def __init__(self, model_name_dir):
		self.model_name_dir = model_name_dir
	def on_epoch_end(self, batch, logs={}):
		pdb.set_trace()
		my_plots.save_model_as_image(self.model, save_path = PATH_IMAGES+model_name_dir, filename_prefix = '', normalize='local', mono=False)

