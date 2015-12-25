'''
24 Dec 2015 (Best day of the year to code), Keunwoo Choi
load all .cp files and put them into hdf with corresponding dataset names.
'''
import os
import sys
import h5py
import numpy as np
import cPickle
import my_utils	
from environments import *
from constants import *

def create_hdf_dataset(filename, dataset_name, file_manager, song_file_inds):
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	segment_selection = cp.load(PATH_DATA + FILE_DICT["segment_selection"]) # track_id : (boundaries, labels)
	clips_per_song = 3
	num_songs = len(song_file_inds)
	num_clips = clips_per_song*num_songs
	# create hdf file.
	file_write = h5py.File(PATH_HDF + filename)
	if dataset_name == 'cqt':
		tf_representation = file_manager.load_cqt(0) # change to more general name than 'tf_represnetation'

	cqt_height, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	cqt_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds
	# create dataset
	data_cqt = file_train.create_dataset(dataset_name, (num_clips, 1, cqt_height, cqt_width), maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	cqt_stereo = np.zeros((cqt_height, cqt_width, 2))
	cqt_downmix = np.zeros((cqt_height, cqt_width, 1))
	# fill the dataset.
	for song_idx in song_file_inds: 
		track_id = track_ids[song_idx]
		if dataset_name == 'cqt':
			cqt_stereo = file_manager.load_cqt(song_idx) # height, width, 2
		#elif stft,, ..
		cqt_downmix = np.zeros(cqt_height, cqt_stereo.shape[1], 1)
		cqt_downmix = cqt_stereo[:,:,0] + cqt_stereo[:,:,1] # height, width

		boundaries = segment_selection[track_id]
		if len(boundaries) < clips_per_song:
			boundaries = []
			num_frames = cqt_downmix.shape()[1]
			for i in xrange(clips_per_song):
				frame_from = (i+1)*num_frames/(clips_per_song+1)
				boundaries.append((frame_from,frame_from+cqt_width))
		for clip_idx in xrange(clips_per_song):
			#for segment_idx in [0]:
			frame_from, frame_to = boundaries[clip_idx] # TODO : ?? [0]? all 3 segments? ??? how??
			cqt_selection = cqt_downmix[:, frame_from:frame:to, :]
		# put this cqt selection into hdf dataset.
			data_cqt[song_idx + clip_idx*num_train_songs, :, :, :] = cqt_selection.transpose((2, 0, 1)) # 1, height, width

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
	
	