import keras


class History(keras.callbacks.Callback):
	"""history, not validation. use History_Val to include both training and validation data"""
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accs = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accs.append(logs.get('acc'))

class History_Val(keras.callbacks.Callback):
	"""history with valudation data"""
	def on_train_begin(self, logs={}):
		self.losses = []
		self.accs = []
		self.val_losses = []
		self.val_accs = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		self.accs.append(logs.get('acc'))
		self.val_losses.append(logs.get('val_loss'))
		self.val_accs.append(logs.get('val_acc'))