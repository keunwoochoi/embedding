""" To predict tags! using ilm10k data, stft or cqt representation, 
"""
#import matplotlib
#matplotlib.use('Agg')
from constants import *
from environments import *
import numpy as np
import librosa
import keras
import os
import pdb
import my_keras_models
import my_keras_utils
import cPickle as cP
import time
import sys
#import my_plots

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

def get_input_output_set(file_manager, indices, truths, type, max_len_freq=256, width_image=256, clips_per_song=0):
	"""indices: list consists of integers between [0, 9320], 
	usually it is one of train_inds, valid_inds, test_inds.
	it returns data_x and data_y.
	file_manager: an instance of File_Manager class.
	type = 'stft' or 'cqt', determines which function file_manager should use
	clips_per_song= integer, 0,1,2,...N: decide how many clips it will take from a song

	"""
	if type=='stft':
		tf_representation = file_manager.load_stft(0)
		len_freq, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.

	elif type=='cqt':
		tf_representation = file_manager.load_cqt(0)
		len_freq, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	if len_freq > max_len_freq:
		len_freq = max_len_freq

	num_labels = truths.shape[1]
	width = width_image
	print '   -- check number of all data --'
	num_data = 0
	if clips_per_song==0:
		for i in indices:
			tf_representation = file_manager.load_stft(i)
			num_data += tf_representation.shape[1] / width
	else:
		num_data = len(indices) * clips_per_song
	print '   -- check:done, num_data is %d --' % num_data

	ret_x = np.zeros((num_data, num_ch, len_freq, width)) # x : 4-dim matrix, num_data - num_channel - height - width
	ret_y = np.zeros((num_data, num_labels)) # y : 2-dum matrix, num_data - labels (or whatever)

	if type not in ['stft', 'cqt']:
		print "wront type in get_input_output_set, so failed to prepare data."

	data_ind = 0
	for i in indices: # for every song
		# print i
		if type == 'stft':
			tf_representation = np.abs(file_manager.load_stft(i))
		elif type=='cqt':
			tf_representation = file_manager.load_cqt(i)

		tf_representation = np.expand_dims(tf_representation[:len_freq, :, :], axis=3) # len_freq, num_fr, num_ch, nothing(#data). -->
		# print 'expending done'
		num_fr = tf_representation.shape[1]
		tf_representation = tf_representation.transpose((3, 2, 0, 1)) # nothing, num_ch, len_freq, num_fr
		#print 'transpose done'
		if clips_per_song == 0:
			for j_ind in xrange(num_fr/len_freq):
				ret_x[data_ind, :, :, :] = tf_representation[:, :, :, j_ind*width: (j_ind+1)*width]
				ret_y[data_ind, :] = np.expand_dims(truths[i,:], axis=1).transpose()
				data_ind += 1
		else:
			for j_in in xrange(clips_per_song):
				frame_from = 43 + j_in*((num_fr-width_image-43*2)/clips_per_song) # remove 1-sec from both ends
				frame_to = frame_from + width_image

				ret_x[data_ind, :, :, :] = tf_representation[:, :, :, frame_from:frame_to]
				ret_y[data_ind, :] = np.expand_dims(truths[i,:], axis=1).transpose()
				data_ind += 1
	return ret_x, ret_y

def load_all_sets(label_matrix, clips_per_song):
	file_manager = File_Manager()

	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=5)
	num_songs_train = min(50, len(train_inds))
	train_inds = train_inds[0:num_songs_train]
	valid_inds = valid_inds[0:40]
	test_inds  = test_inds [0:40]
	print "--- Lets go! ---"
	start = time.clock()
	train_x, train_y = get_input_output_set(file_manager, train_inds, truths=label_matrix, type='stft', max_len_freq=256, width_image=256, clips_per_song=clips_per_song)
	until = time.clock()
	print "--- train data prepared; %d clips from %d songs, took %d seconds to load---" % (len(train_x), len(train_inds), (until-start) )
	start = time.clock()
	valid_x, valid_y = get_input_output_set(file_manager, valid_inds, truths=label_matrix, type='stft', max_len_freq=256, width_image=256, clips_per_song=clips_per_song)
	until = time.clock()
	print "--- valid data prepared; %d clips from %d songs, took %d seconds to load---" % (len(valid_x), len(valid_inds), (until-start) )
	start = time.clock()
	test_x,  test_y  = get_input_output_set(file_manager, test_inds, truths=label_matrix, type='stft', max_len_freq=256, width_image=256, clips_per_song=clips_per_song)
	until = time.clock()
	print "--- test data prepared; %d clips from %d songs, took %d seconds to load---" % (len(test_x), len(test_inds), (until-start) )

	return train_x, train_y, valid_x, valid_y, test_x, test_y

def print_usage_and_die():
	print 'python filename num_of_epoch(integer)'
	print 'ex) $ python main_learn_tag.py 40'
	sys.exit()

if __name__ == "__main__":
	# if len(sys.argv) < 2:
	# 	print_usage_and_die

	# nb_epoch = int(sys.argv[1])

	nb_epoch = 200
	clips_per_song = 5
	# label matrix
	dim_latent_feature = 10
	# label_matrix_filename = (FILE_DICT["mood_latent_matrix"] % dim_latent_feature)
	label_matrix_filename = (FILE_DICT["mood_latent_tfidf_matrix"] % dim_latent_feature) # tfidf is better!
	
	if os.path.exists(PATH_DATA + label_matrix_filename):
		label_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
	else:
		"print let's cook the mood-latent feature matrix"
		import main_prepare
		mood_tags_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
		label_matrix = main_prepare.get_LDA(X=mood_tags_matrix, num_components=k, show_topics=False)
		np.save(PATH_DATA + label_matrix_filename, W)
	print 'size of mood tag matrix:'
	print label_matrix.shape

	# load dataset
	print "I'll take %d clips for each song." % clips_per_song
	train_x, train_y, valid_x, valid_y, test_x, test_y = load_all_sets(label_matrix=label_matrix, clips_per_song=clips_per_song)
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100

	#prepare model
	model_name = 'test_model_latent_10_tfidf_4lyers'

	start = time.clock()
	print "--- going to build a keras model with height:%d, width:%d, num_labels:%d" % (train_x.shape[2], train_x.shape[3], train_y.shape[1])
 	model = my_keras_models.build_convnet_model(height=train_x.shape[2], width=train_x.shape[3], num_labels=train_y.shape[1])
 	until = time.clock()
 	print "--- keras model was built, took %d seconds ---" % (until-start)

	#prepare callbacks
	history = my_keras_utils.History_Val()
	#train!
	model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=40, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, callbacks=[history])
	# score = model.evaluate(test_x, test_y, batch_size=batch_size, show_accuracy=True, verbose=1)
	model.evaluate(test_x, test_y, show_accuracy=True)
	# model.save_weights(PATH_MODEL + model_name + '_after_60.keras') # fix h5py simbolic link error.
	#   which is, ImportError: libhdf5.so.8: cannot open shared object file: No such file or directory

	print history.losses
	print history.accs
	print history.val_losses
	print history.val_accs
	# figure_filepath = PATH_FIGURE + model_name + '_history.png'
	# my_plots.export_history(history.accs, history.val_accs, history.losses, history.val_losses, figure_filepath, net_name=None)

