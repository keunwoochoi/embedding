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


if __name__=="__main__":

	file_manager = my_utils.File_Manager()

	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=10)
	train_inds = train_inds[:10] # for test
	
	segment_selection = cp.load(PATH_DATA + FILE_DICT["segment_selection"]) # track_id : (boundaries, labels)
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	
	num_train = len(train_inds)
	num_valid = len(valid_inds)
	num_test  = len(test_inds)

	file_train = h5py.File(PATH_HDF + 'data_train.h5')
	file_valid = h5py.File(PATH_HDF + 'data_valid.h5')
	file_test  = h5py.File(PATH_HDF + 'data_test.h5')
	#for each file,
	# and if train, for each features.
	tf_type = 'cqt'
	tf_representation = file_manager.load_cqt(0)
	cqt_height, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	cqt_width = int(6 * (CQT_CONST["sr"] / CQT_CONST["hop_len"])) # 6-seconds

	data_cqt_1 = file_train.create_dataset("cqt_1", (num_train, 1, cqt_height, cqt_width), maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	cqt_stereo = np.zeros((cqt_height, cqt_width, 2))
	cqt_downmix = np.zeros((cqt_height, cqt_width, 1))


	for idx in train_inds: 
		track_id = track_ids[idx]
		cqt_stereo = file_manager.load_cqt(idx) # height, width, 2
		cqt_downmix = np.zeros(cqt_height, cqt_stereo.shape[1], 1)
		cqt_downmix = cqt_stereo[:,:,0] + cqt_stereo[:,:,1] # height, width
		print 'put here: select clip from the song. '
		boundaries = segment_selection[track_id]
		#for segment_idx in [0]:
		frame_from, frame_to = boundaries[0]
		cqt_selection = cqt_downmix[:, frame_from:frame:to, :]
		# put this cqt selection into hdf dataset.
		data_cqt_1[idx, :, :, :] = cqt_downmix.transpose((2, 0, 1)) # 1, height, width
		
	file_train.close()
	