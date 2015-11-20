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
import time

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
	if type=='stft':
		tf_representation = file_manager.load_stft(0)
		len_freq, num_fr, num_ch = tf_representation.shape # 513, 6721, 2 for example.

	elif type=='cqt':
		tf_representation = file_manager.load_cqt(0)
		len_freq, num_fr, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	
	num_labels = truths.shape[1]
	width = len_freq

	ret_x = np.zeros((0, num_ch, len_freq, width)) # x : 4-dim matrix, num_data - num_channel - height - width
	ret_y = np.zeros((0, num_labels)) # y : 2-dum matrix, num_data - labels (or whatever)

	if type not in ['stft', 'cqt']:
		print "wront type in get_input_output_set, so failed to prepare data."

	for i in indices:
		if type == 'stft':
			tf_representation = file_manager.load_stft(i)
		elif type=='cqt':
			tf_representation = file_manager.load_cqt(i)

		tf_representation = np.expand_dims(tf_representation, axis=3) # len_freq, num_fr, num_ch, nothing(#data). -->
		tf_representation = tf_representation.transpose((3, 2, 0, 1)) # nothing, num_ch, len_freq, num_fr
		
		for j in xrange(num_fr/len_freq):
			ret_x = np.concatenate((ret_x, tf_representation[:,:, :, j*width: (j+1)*width]), axis=0)
			pdb.set_trace()
			ret_y = np.concatenate((ret_y, truths[i,:]), axis=0)

	return ret_x, ret_y

if __name__ == "__main__":

	mood_tags_matrix = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	print 'size of mood tag matrix:'
	print mood_tags_matrix.shape

	file_manager = File_Manager()

	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=5)
	train_inds = train_inds[0:30]
	valid_inds = valid_inds[0:15]
	test_inds  = test_inds [0:15]
	
	start = time.clock()
	train_x, train_y = get_input_output_set(file_manager, train_inds, mood_tags_matrix, 'stft')
	until = time.clock()
	print "--- train data prepared; %d clips from %d songs, took %d seconds to load---" % (len(train_x), len(train_inds), (until-start) )
	start = time.clock()
	valid_x, valid_y = get_input_output_set(file_manager, valid_inds, mood_tags_matrix, 'stft')
	until = time.clock()
	print "--- valid data prepared; %d clips from %d songs, took %d seconds to load---" % (len(valid_x), len(valid_inds), (until-start) )
	start = time.clock()
	test_x,  test_y  = get_input_output_set(file_manager, test_inds, mood_tags_matrix, 'stft')
	until = time.clock()
	print "--- test data prepared; %d clips from %d songs, took %d seconds to load---" % (len(test_x), len(test_inds), (until-start) )
	start = time.clock()
	len(train_y[0])
	pdb.set_trace()
	model = my_keras_models.build_convnet_model(height=train_x[0].shape[0], width=train_x[0].shape[1], num_labels=len(train_y[0]))
	until = time.clock()
	print "--- keras model was built, took %d seconds ---" % (until-start)
	pdb.set_trace()
	model.fit(train_x, train_y, batch_size=32, nb_epoch=40, validation_data=(valid_x, valid_y), show_accuracy=True, verbose=1)
	
	model.fit(train_x, train_y, nb_epoch=40)
	# score = model.evaluate(test_x, test_y, batch_size=batch_size, show_accuracy=True, verbose=1)
	model.evaluate(test_x, test_yshow_accuracy=True)






