'''
24 Dec 2015 (Best day of the year to code), Keunwoo Choi
load all .cp files and put them into hdf with corresponding dataset names.
'''
import os
import sys
import h5py
import numpy as np
import cPickle as cP
import itertools
from multiprocessing import Pool
import my_utils	
from environments import *
from constants import *
import pdb

FILE_MANAGER = my_utils.File_Manager()

def select_and_save_each(args):
	track_id, idx, boundaries, path, tf_type = args
	if os.path.exists(path+str(track_id)+'.npy'):
		print 'track_id %d: already done.' % track_id
		return
	clips_per_song = 3
	tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	tf_stereo = FILE_MANAGER.load(ind=idx, data_type=tf_type) # height, width, 2
	
	if len(boundaries) < clips_per_song:
		boundaries = []
		num_frames = tf_stereo.shape[1]
		for i in xrange(clips_per_song):
			frame_from = (i+1)*num_frames/(clips_per_song+1)
			boundaries.append((frame_from,frame_from+tf_width))
	for clip_idx in xrange(clips_per_song):
		#for segment_idx in [0]:
		frame_from, frame_to = boundaries[clip_idx] # TODO : ?? [0]? all 3 segments? ??? how??
		frame_to = frame_from + tf_width
		if frame_to > tf_stereo.shape[1]:
			frame_to = tf_stereo.shape[1]
			frame_from = frame_to - tf_width
		if tf_type =='cqt':
			tf_selection = my_utils.inv_log_amplitude(tf_stereo[:, frame_from:frame_to, 0]) + \
							my_utils.inv_log_amplitude(tf_stereo[:, frame_from:frame_to, 1])
		elif tf_type =='stft':
			tf_selection = np.abs(tf_stereo[:, frame_from:frame_to, 0]) + \
							np.abs(tf_stereo[:, frame_from:frame_to, 1])
	np.save(path+str(track_id)+'.npy' , my_utils.log_amplitude(tf_selection))
	print 'track_id %d: done.' % track_id
	return

def select_and_save(tf_type):
	'''select and save cqt and stft using multiprocessing
	for CQT and STFT.'''
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	idx = range(len(track_ids))
	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	segment_selection_list = [segment_selection[key] for key in track_ids]

	path = PATH_HDF + 'temp_' + tf_type + '/'
	path_list = [path]*len(track_ids)
	tf_type_list = [tf_type]*len(track_ids)

	args = zip(track_ids, idx, segment_selection_list, path_list, tf_type_list)
	if not os.path.exists(path):
		os.mkdir(path)

	p = Pool(24)
	p.map(select_and_save_each, args)


def create_hdf_dataset(filename, dataset_name, file_manager, song_file_inds):
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	clips_per_song = 3
	num_songs = len(song_file_inds)
	num_clips = clips_per_song*num_songs

	# get the size of dataset.
	if dataset_name in ['cqt', 'stft']:
		tf_representation = file_manager.load(ind=0, data_type=dataset_name) # change to more general name than 'tf_represnetation'
	tf_height, num_fr_temp, num_ch = tf_representation.shape # 513, 6721, 2 for example.
	tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	tf_stereo = np.zeros((tf_height, tf_width, 2))

	# create or load dataset
	if os.path.exists(PATH_HDF + filename):
		file_write = h5py.File(PATH_HDF + filename, 'r+')
		print 'loading hdf file that exists already there.'
	else:
		file_write = h5py.File(PATH_HDF + filename, 'w')
		print 'creating new hdf file.'
	if dataset_name in file_write:
		data_cqt = file_write[dataset_name]
	else:
		data_cqt = file_write.create_dataset(dataset_name, (num_clips, 1, tf_height, tf_width), maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	# create or load 'log' file for this work. -- to resume easily.
	if os.path.exists(PATH_HDF + 'log_for_' + filename):
		idx_until = np.load(PATH_HDF + 'log_for_' + filename)
	else:
		idx_until = 0
	
	# fill the dataset.
	for song_idx, track_id in enumerate(song_file_inds):
		if song_idx < idx_until:
			print 'idx %d is already done, so skipp this.' % song_idx
			continue
		track_id = track_ids[song_idx]
		if dataset_name in ['cqt', 'stft']:
			tf_stereo = file_manager.load(ind=song_idx, data_type=dataset_name) # height, width, 2
			if dataset_name == 'stft':
				tf_stereo = np.abs(tf_stereo)
			elif dataset_name=='cqt':
				tf_stereo = my_utils.inv_log_amplitude(tf_stereo) # decibel to linear
		else:
			print 'not ready for other types of data.'
			return

		#tf_downmix = np.expand_dims(tf_downmix, axis=2)
		boundaries = segment_selection[track_id]
		if len(boundaries) < clips_per_song:
			boundaries = []
			num_frames = tf_stereo.shape[1]
			for i in xrange(clips_per_song):
				frame_from = (i+1)*num_frames/(clips_per_song+1)
				boundaries.append((frame_from,frame_from+tf_width))
		for clip_idx in xrange(clips_per_song):
			#for segment_idx in [0]:
			frame_from, frame_to = boundaries[clip_idx] # TODO : ?? [0]? all 3 segments? ??? how??
			frame_to = frame_from + tf_width
			if frame_to > tf_stereo.shape[1]:
				frame_to = tf_stereo.shape[1]
				frame_from = frame_to - tf_width
			if dataset_name=='cqt':
				tf_selection = my_utils.inv_log_amplitude(tf_stereo[:, frame_from:frame_to, 0]) + \
								my_utils.inv_log_amplitude(tf_stereo[:, frame_from:frame_to, 1])
			elif dataset_name =='stft':
				tf_selection = np.abs(tf_stereo[:, frame_from:frame_to, 0]) + \
								np.abs(tf_stereo[:, frame_from:frame_to, 1])

		# put this cqt selection into hdf dataset.
			data_cqt[song_idx + clip_idx*num_songs, 0, :, :] = my_utils.log_amplitude(tf_selection) # 1, height, width
		
		if song_idx % 10 == 0:
			np.save(PATH_HDF + 'log_for_' + filename, song_idx)
		print 'Done: cp2hdf, song_idx:%d, track_id: %d' % (song_idx, track_id)

		


	file_write.close()
	return

if __name__=="__main__":

	datatype = sys.argv[1] #'cqt, stft' 
	print 'datatype: %s' % datatype

	select_and_save(datatype)
	sys.exit(0)

	# after create all file for cqt and stft with selected segments, then add them on hdf.
	file_manager = my_utils.File_Manager()
	train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=10)
	#train_inds = train_inds[:10] # for test
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	
	create_hdf_dataset(filename='data_train.h5', 
						dataset_name=datatype,
						file_manager=file_manager,
						song_file_inds=train_inds)
	#	num_valid = len(valid_inds)
	#	num_test  = len(test_inds)

	#file_train = h5py.File(PATH_HDF + 'data_train.h5')
	# file_valid = h5py.File(PATH_HDF + 'data_valid.h5')
	# file_test  = h5py.File(PATH_HDF + 'data_test.h5')
	
	#for each file,
	# and if train, for each features.
	
	