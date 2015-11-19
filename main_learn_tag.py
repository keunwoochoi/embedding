""" To predict tags! using ilm10k data, stft or cqt representation, 
"""
from constants import *
from environments import *
import numpy as np
import librosa
import keras
import os
import pdb
import my_keras_models
import cPickle as cP

class File_Manager():
	def __init__(self):
		self.track_ids = cP.load(open(PATH_DATA + FILE_DICT["track_ids"], 'r')) #list, 9320
		self.id_path = cP.load(open(PATH_DATA + FILE_DICT["id_path"], 'r')) #dict, 9320
		self.filenum = len(self.track_ids)
		print "file manager init with %d track ids and %d element dictionary " % (self.filenum, len(self.id_path))

	def load_src(self, ind):
		if ind > len(self.track_ids):
			print 'wrong ind -- too large: %d' % ind
		path = self.id_path[self.track_ids[ind]]
		return librosa.load(path, sr=SR, mono=False)

	def load_stft(self, ind):
		return np.load( PATH_STFT + str(self.track_ids[ind]) + '.npy')

	def load_cqt(self, ind):
		return np.load( PATH_CQT + str(self.track_ids[ind]) + '.npy')

	def split_inds(self, num_folds):
		"""returns index of train/valid/test songs"""
		if num_folds < 3:
			return "wrong num_folds, should be >= 3"
		num_test = self.filenum / num_folds
		num_valid = self.filenum / num_folds
		num_train = self.filenum - (num_test + num_valid)

		rand_filename = PATH_DATA +("random_permutation_%d_%d.npy" % (num_folds, self.filenum))
		if os.path.exists(rand_filename):
			rand_inds = np.load(rand_filename)
		else:
			rand_inds = np.random.permutation(self.filenum)
			np.save(rand_filename, rand_inds)

		return rand_inds[0:num_train], rand_inds[num_train:num_train+num_valid], rand_inds[num_train+num_valid:]

def get_input_output_set(file_manager, indices, truths, type):
	"""indices: list consists of integers between [0, 9320], 
	usually it is one of train_inds, valid_inds, test_inds.
	it returns data_x and data_y.
	file_manager: an instance of File_Manager class.
	type = 'stft' or 'cqt', determines which function file_manager should use

	"""
	ret_x = []
	ret_y = []
	if type not in ['stft', 'cqt']:
		print "wront type in get_input_output_set, so failed to prepare data."

	for i in indices:
		if type == 'stft':
			tf_representation = file_manager.load_stft(i)
		elif type=='cqt':
			tf_representation = file_manager.load_stft(i)

		len_freq, num_fr, num_ch = tf_representation.shape # 513, 6721, 2 for example.
		width = len_freq
		for j in xrange(num_fr/len_freq):
			ret_x.append(tf_representation[:, j*width: (j+1)*width, :])
			ret_y.append(truths[i,:])

	return ret_x, ret_y

if __name__ == "__main__":

	mood_tags_matrix = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	print 'size of mood tag matrix:'
	print mood_tags_matrix.shape

	file_manager = File_Manager()

	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=5)
	train_inds = train_inds[0:1]
	valid_inds = valid_inds[0:1]
	test_inds  = test_inds [0:1]
	train_x, train_y = get_input_output_set(file_manager, train_inds, mood_tags_matrix, 'stft')
	print "--- train data prepared; %d ---" % len(train_x)
	valid_x, valid_y = get_input_output_set(file_manager, valid_inds, mood_tags_matrix, 'stft')
	print "--- valid data prepared: %d ---" % len(valid_x)
	test_x,  test_y  = get_input_output_set(file_manager, test_inds, mood_tags_matrix, 'stft')
	print "--- test data prepared:  %d ---" % len(test_x)

	model = my_keras_models.build_convnet_model(height=train_x[0].shape[0], width=train_x[0].shape[1], num_labels=len(train_y[0]))
	model.fit(train_x, train_y, nb_epoch=20, show_accuracy=True, verbose=1)

	# score = model.evaluate(test_x, test_y, batch_size=batch_size, show_accuracy=True, verbose=1)







