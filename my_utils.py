
from constants import *
from environments import *
from training_settings import *

import os
import cPickle as cP
import time
import sys
import numpy as np
import adjspecies
import pprint	

class Hyperparams_Manager():
	def __init__(self):
		self.dict = {} #key:adjspecies, value: setting_dict
		self.dict_name2str = {} #key: ajdspecies, value: str(setting_dict.values())
		self.dict_str2name = {} # reverse of above.

	def check_and_save(self, setting_dict):
		if not has_setting(setting_dict):
			self.save_new_setting(setting_dict)
		else:
			print 'This setting dictionary is already stored.'

	def get_name(self, setting_dict):
		'''return new name if it is new.
		return previously used name if it is the same again.
		'''
		if self.has_setting(setting_dict):
			print 'return once used name:%s' %  self.dict_str2name[dict2str(setting_dict)]
			return self.dict_str2name[dict2str(setting_dict)]
		else:
			return self.pick_new_name()

	def pick_new_name(self):
		new_name = adjspecies.random_adjspecies(sep='_', maxlen=10, prevent_stutter=True)
		while new_name in self.dict:
			new_name = adjspecies.random_adjspecies(sep='_', maxlen=10, prevent_stutter=True)
		return new_name

	def save_new_setting(self, setting_dict):	
		new_name = self.pick_new_name()	
		self.dict[new_name] = setting_dict
		long_name = dict2str(setting_dict)
		self.dict_name2str[new_name]= long_name
		self.dict_str2name[long_name] = new_name
		cP.dump(self, open(PATH_DATA + FILE_DICT["hyperparam_manager"], 'w'))

	def list_setting_names(self):
		pprint.pprint(self.dict.keys())

	def list_settings(self):
		pprint.pprint(self.dict)

	def has_setting(self, setting_dict):
		return dict2str(setting_dict) in self.dict_str2name

def dict2str(setting_dict):
	return '_'.join([key+'.'+str(setting_dict[key]) for key in setting_dict])

class File_Manager():
	def __init__(self):
		self.track_ids = cP.load(open(PATH_DATA + FILE_DICT["track_ids"], 'r')) #list, 9320
		self.id_path = cP.load(open(PATH_DATA + FILE_DICT["id_path"], 'r')) #dict, 9320
		self.filenum = len(self.track_ids)
		print "file manager init with %d track ids and %d element dictionary " % (self.filenum, len(self.id_path))
	'''
	def load_src(self, ind):
		import librosa
		if ind > len(self.track_ids):
			print 'wrong ind -- too large: %d' % ind
		path = self.id_path[self.track_ids[ind]]
		return librosa.load(path, sr=SR, mono=False)
	'''
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

def write_setting_as_texts(path_to_save, setting_dict):
	from datetime import datetime
	timeinfo = datetime.now().strftime('%Y-%m-%d %H:%M')
	f = open(path_to_save + timeinfo + '.time', 'w')
	f.close()
	for key in setting_dict:
		with open(path_to_save+ 'a_'+str(key)+ ': ' + str(setting_dict[key])+'.txt', 'w') as f:
			pass
	return

def get_input_output_set(file_manager, indices, truths, tf_type, max_len_freq=256, width_image=256, clips_per_song=0):
	"""indices: list consists of integers between [0, 9320], 
	usually it is one of train_inds, valid_inds, test_inds.
	it returns data_x and data_y.
	file_manager: an instance of File_Manager class.
	type = 'stft' or 'cqt', determines which function file_manager should use
	clips_per_song= integer, 0,1,2,...N: decide how many clips it will take from a song

	"""
	# first, set the numbers
	if tf_type=='stft':
		tf_representation = file_manager.load_stft(0)
		len_freq, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.

	elif tf_type=='cqt':
		tf_representation = file_manager.load_cqt(0)
		len_freq, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	if len_freq > max_len_freq:
		len_freq = max_len_freq
	else:
		print 'You set max_len_freq as %d, but it doesnt have that many frequency bins, so it will use all it has, which is %d.' % (max_len_freq, len_freq)

	num_labels = truths.shape[1]
	width = width_image
	print '   -- check number of all data --'
	num_data = 0
	if clips_per_song==0:
		for i in indices:
			if tf_type=='stft':
				tf_representation = file_manager.load_stft(i)
			elif tf_type=='cqt':
				tf_representation = file_manager.load_cqt(i)
			num_data += tf_representation.shape[1] / width
	else:
		num_data = len(indices) * clips_per_song
	print '   -- check:done, num_data is %d --' % num_data

	ret_x = np.zeros((num_data, num_ch, len_freq, width)) # x : 4-dim matrix, num_data - num_channel - height - width
	ret_y = np.zeros((num_data, num_labels)) # y : 2-dum matrix, num_data - labels (or whatever)

	if tf_type not in ['stft', 'cqt']:
		print "wront type in get_input_output_set, so failed to prepare data."

	data_ind = 0
	for i in indices: # for every song
		# print i
		if tf_type == 'stft':
			tf_representation = 10*np.log10(np.abs(file_manager.load_stft(i)))
		elif tf_type=='cqt':
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
				frame_from = 43*10 + j_in*((num_fr-width_image-43*10*2)/clips_per_song) # remove 1-sec from both ends
				frame_to = frame_from + width_image

				ret_x[data_ind, :, :, :] = tf_representation[:, :, :, frame_from:frame_to]
				ret_y[data_ind, :] = np.expand_dims(truths[i,:], axis=1).transpose()
				data_ind += 1
	return ret_x, ret_y

def load_all_sets(label_matrix, clips_per_song, num_train_songs=100, tf_type=None):
	if not tf_type:
		print '--- tf_type not specified, so stft is assumed. ---'
		tf_type = 'stft'
	if tf_type not in ['stft', 'cqt']:
		print '--- wrong tf_type:%s, it should be either stft or cqt---' % tf_type
		return

	file_manager = File_Manager()

	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=5)
	num_songs_train = min(num_train_songs, len(train_inds))
	
	train_inds = train_inds[0:num_songs_train]
	valid_inds = valid_inds[:400]
	test_inds  = test_inds [:500]
	print "--- Lets go! ---"
	start = time.time()
	train_x, train_y = get_input_output_set(file_manager, train_inds, truths=label_matrix,
	 										tf_type=tf_type, 
	 										max_len_freq=TR_CONST["height_image"], 
	 										width_image=TR_CONST["width_image"], 
	 										clips_per_song=TR_CONST["clips_per_song"])
	until = time.time()
	print "--- train data prepared; %d clips from %d songs, took %d seconds to load---" \
									% (len(train_x), len(train_inds), (until-start) )
	start = time.time()
	valid_x, valid_y = get_input_output_set(file_manager, valid_inds, truths=label_matrix, 
											tf_type=tf_type, 
											max_len_freq=TR_CONST["height_image"], 
											width_image=TR_CONST["width_image"], 
											clips_per_song=TR_CONST["clips_per_song"])
	until = time.time()
	print "--- valid data prepared; %d clips from %d songs, took %d seconds to load---" \
									% (len(valid_x), len(valid_inds), (until-start) )
	start = time.time()
	test_x,  test_y  = get_input_output_set(file_manager, test_inds, truths=label_matrix, 
											tf_type=tf_type, 
											max_len_freq=TR_CONST["height_image"], 
											width_image=TR_CONST["width_image"], 
											clips_per_song=TR_CONST["clips_per_song"])
	until = time.time()
	print "--- test data prepared; %d clips from %d songs, took %d seconds to load---" \
									% (len(test_x), len(test_inds), (until-start) )
	
	if tf_type == 'cqt':
		global_mean = -61.25 # computed from the whole data for cqt
		global_std  = 14.36
	elif tf_type == 'stft':
		global_mean = -61.25 # should be mended with STFT values
		global_std  = 14.36

	train_x = (train_x - global_mean)/global_std	
	valid_x = (valid_x - global_mean)/global_std
	test_x  = (test_x - global_mean) /global_std

	return train_x, train_y, valid_x, valid_y, test_x, test_y


