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
import time

FILE_MANAGER = my_utils.File_Manager()
HEIGHT = {}
HEIGHT['cqt'] = FILE_MANAGER.load_cqt(0).shape[0]
HEIGHT['stft']= FILE_MANAGER.load_stft(0).shape[0]
print 'cqt and stft height: %d and %d' % (HEIGHT['cqt'], HEIGHT['stft'])

def select_and_save_each(args):
	track_id, idx, boundaries, path, tf_type = args
	print 'idx, track_id: %d, %d, start!' % (idx, track_id)
	if os.path.exists(path+str(track_id)+'.npy'):
		print '%d, track_id %d: already done.' % (idx, track_id)
		return
	# pdb.set_trace()
	clips_per_song = 3
	tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	tf_stereo = FILE_MANAGER.load(ind=idx, data_type=tf_type) # height, width, 2
	ret = np.zeros((HEIGHT[tf_type], tf_width, clips_per_song))
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
		ret[:,:,clip_idx] = my_utils.log_amplitude(tf_selection)
	np.save(path+str(track_id)+'.npy' , ret)
	print '%d, track_id %d: done.' % (idx, track_id)
	return

def select_and_save(tf_type):
	'''select and save cqt and stft using multiprocessing
	for CQT and STFT.
	Do this, and then execute create_hdf_dataset()'''
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

	# print 'DEBUGGING'
	# for arg in args:
	# 	select_and_save_each(arg)
	p = Pool(48)
	p.map(select_and_save_each, args)


def create_hdf_dataset(filename, dataset_name, file_manager, song_file_inds):
	'''filename: .h5 filename to store.
	dataset_name: e.g. 'cqt', 'stft', i.e. key of the h5 file.
	song_file_inds: index <= 9320.
	'''
	print 'create_hdf_dataset begins.'
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	clips_per_song = 3
	num_songs = len(song_file_inds)
	num_clips = clips_per_song*num_songs
	print 'num_songs:%d, num_clips:%d' % (num_songs, num_clips)
	# get the size of dataset.
	if dataset_name in ['cqt', 'stft']:
		tf_representation = file_manager.load(ind=0, data_type=dataset_name) # change to more general name than 'tf_represnetation'
		tf_height = HEIGHT[dataset_name]
	tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	
	path = PATH_HDF + 'temp_' + dataset_name + '/' # path to read numpy files

	# create or load dataset
	if os.path.exists(PATH_HDF_TEMP + filename):
		file_write = h5py.File(PATH_HDF_TEMP + filename, 'r+')
		print 'loading hdf file that exists already there.'
	else:
		file_write = h5py.File(PATH_HDF_TEMP + filename, 'w')
		print 'creating new hdf file.'
	if dataset_name in file_write:
		data_cqt = file_write[dataset_name]
	else:
		data_cqt = file_write.create_dataset(dataset_name, (num_clips, 1, tf_height, tf_width), maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	# fill the dataset.
	done_idx = np.load(PATH_HDF_TEMP + filename + dataset_name + '_done_idx.npy')
	for dataset_idx in xrange(len(song_file_inds)):
		if dataset_idx <= done_idx:
			continue			
		song_idx = song_file_inds[dataset_idx]
		track_id = track_ids[song_idx]
		# put this cqt selection into hdf dataset.
		tf_selections = np.load(path + str(track_id) + '.npy')
		for clip_idx in range(clips_per_song):
			data_cqt[dataset_idx + clip_idx*num_songs, 0, :, :] =  tf_selections[:,:,clip_idx]
		print 'Done: cp2hdf, dataset_idx:%d, track_id: %d' % (dataset_idx, track_id)
		np.save(PATH_HDF_TEMP + filename + dataset_name + '_done_idx.npy', dataset_idx)

	print ' ======== it is all done for %s! ========' % dataset_name
	file_write.close()
	return

if __name__=="__main__":

	datatype = sys.argv[1] #'cqt, stft' 
	worktype = sys.argv[2] #a, b

	print 'datatype: %s' % datatype
	if worktype == 'a':
		select_and_save(datatype)
		sys.exit(0)
	elif worktype == 'b':
	# after create all file for cqt and stft with selected segments, then add them on hdf.
		file_manager = my_utils.File_Manager()
		train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=10)
	
		create_hdf_dataset(filename='data_train.h5', 
							dataset_name=datatype,
							file_manager=file_manager,
							song_file_inds=train_inds)
	
	
	