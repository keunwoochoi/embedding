import keras.callbacks

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

