'''
24 Dec 2015 (Best day of the year to code), Keunwoo Choi
load all .cp files and put them into hdf with corresponding dataset names.
'''
import os
import sys
import h5py
import numpy as np
import cPickle as cP
import my_utils	
from environments import *
from constants import *

def create_hdf_dataset(filename, dataset_name, file_manager, song_file_inds):
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	clips_per_song = 3
	num_songs = len(song_file_inds)
	num_clips = clips_per_song*num_songs
	# create hdf file.
	file_write = h5py.File(PATH_HDF + filename)
	if dataset_name == 'cqt':
		tf_representation = file_manager.load_cqt(0) # change to more general name than 'tf_represnetation'

	tf_height, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds
	# create dataset
	data_cqt = file_write.create_dataset(dataset_name, (num_clips, 1, tf_height, tf_width), maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	tf_stereo = np.zeros((tf_height, tf_width, 2))
	tf_downmix = np.zeros((tf_height, tf_width, 1))
	# fill the dataset.
	for song_idx in song_file_inds: 
		track_id = track_ids[song_idx]
		if dataset_name in ['cqt', 'stft']:
			tf_stereo = file_manager.load(ind=song_idx, data_type='dataset_name') # height, width, 2
		else:
			print 'not ready for other types of data.'
			return
		tf_downmix = np.zeros(tf_height, tf_stereo.shape[1], 1)
		tf_downmix = tf_stereo[:,:,0] + tf_stereo[:,:,1] # height, width

		boundaries = segment_selection[track_id]
		if len(boundaries) < clips_per_song:
			boundaries = []
			num_frames = tf_downmix.shape()[1]
			for i in xrange(clips_per_song):
				frame_from = (i+1)*num_frames/(clips_per_song+1)
				boundaries.append((frame_from,frame_from+tf_width))
		for clip_idx in xrange(clips_per_song):
			#for segment_idx in [0]:
			frame_from, frame_to = boundaries[clip_idx] # TODO : ?? [0]? all 3 segments? ??? how??
			tf_selection = tf_downmix[:, frame_from:frame:to, :]
		# put this cqt selection into hdf dataset.
			data_cqt[song_idx + clip_idx*num_train_songs, :, :, :] = tf_selection.transpose((2, 0, 1)) # 1, height, width

	file_write.close()
	return

if __name__=="__main__":

	file_manager = my_utils.File_Manager()
	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=10)
	train_inds = train_inds[:10] # for test
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	
	create_hdf_dataset(filename='data_train.h5', 
						dataset_name='cqt',
						file_manager=file_manager,
						song_file_inds=train_inds)
	#	num_valid = len(valid_inds)
	#	num_test  = len(test_inds)

	file_train = h5py.File(PATH_HDF + 'data_train.h5')
	# file_valid = h5py.File(PATH_HDF + 'data_valid.h5')
	# file_test  = h5py.File(PATH_HDF + 'data_test.h5')
	
	#for each file,
	# and if train, for each features.
	
	