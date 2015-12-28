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
HEIGHT['mfcc']= 19*3
HEIGHT['chroma']=12*3
print 'cqt and stft height: %d and %d' % (HEIGHT['cqt'], HEIGHT['stft'])

def select_and_save_each(args):
	track_id, idx, boundaries, path, tf_type = args
	print 'idx, track_id: %d, %d, start!' % (idx, track_id)
	if os.path.exists(path+str(track_id)+'.npy'):
		print '%d, track_id %d: already done.' % (idx, track_id)
		return
	# pdb.set_trace()
	clips_per_song = 3
	if tf_type in ['cqt', 'stft', 'chroma']:
		tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	elif tf_type in ['mfcc']:
		tf_width = int(6 * MFCC_CONST["frames_per_sec"])

	#load data for whole signal
	if tf_type in ['cqt', 'stft']:
		tf_stereo = FILE_MANAGER.load(ind=idx, data_type=tf_type) # height, width, 2
		num_frames = tf_stereo.shape[1]
	elif tf_type in ['mfcc', 'chroma']:
		tf_triple = FILE_MANAGER.load(ind=idx, data_type=tf_type) # height, width, 3 (for l, r, downmix)
		num_frames = tf_triple.shape[1]

	ret = np.zeros((HEIGHT[tf_type], tf_width, clips_per_song))
	
	if len(boundaries) < clips_per_song:
		boundaries = []
		for i in xrange(clips_per_song):
			frame_from = (i+1)*num_frames/(clips_per_song+1)
			boundaries.append((frame_from,frame_from+tf_width))
	for clip_idx in xrange(clips_per_song):
		#for segment_idx in [0]:
		frame_from, frame_to = boundaries[clip_idx] # TODO : ?? [0]? all 3 segments? ??? how??
		frame_to = frame_from + tf_width
		if frame_to > num_frames:
			frame_to = num_frames
			frame_from = frame_to - tf_width
		
		if tf_type in ['mfcc', 'chroma']:
			tf_selection = tf_triple[:, frame_from:frame_to, 2]
			ret[:,:,clip_idx] = tf_selection # mfcc/chroma --> put directly
		else:
			if tf_type =='cqt': # cqt: inv_log_amp for sum, then log_amp 
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
	for CQT and STFT...and MFCC and chroma
	Do this, and then execute create_hdf_dataset()'''
	if tf_type not in ['stft','cqt','mfcc','chroma','label']:
		raise RuntimeError('Wrong data type, %s.' % tf_type)
		if tf_type == 'label':
			raise RuntimeError('Dont do this for labels. just use FILE_DICT["latent_matrix_w_blahblah.."]')
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	idx = range(len(track_ids))

	path = PATH_HDF + 'temp_' + tf_type + '/'
	if not os.path.exists(path):
		os.mkdir(path)

	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	segment_selection_list = [segment_selection[key] for key in track_ids]

	path_list = [path]*len(track_ids)
	tf_type_list = [tf_type]*len(track_ids)

	args = zip(track_ids, idx, segment_selection_list, path_list, tf_type_list)
	
	# print 'DEBUGGING'
	# for arg in args:
	# 	select_and_save_each(arg)
	p = Pool(48)
	p.map(select_and_save_each, args)


def create_hdf_dataset(filename, dataset_name, file_manager, song_file_inds):
	'''filename: .h5 filename to store.
	dataset_name: e.g. 'cqt', 'stft', 'mfcc', 'chroma', i.e. key of the h5 file.
	song_file_inds: index <= 9320.
	'''
	print 'create_hdf_dataset begins - for filename:%s, dataset_name:%s, %d files' % \
										(filename, dataset_name, len(song_file_inds))
	
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], "r")) # track_id : (boundaries, labels)
	clips_per_song = 3
	num_songs = len(song_file_inds)
	num_clips = clips_per_song*num_songs
	print 'num_songs:%d, num_clips:%d' % (num_songs, num_clips)
	# get the size of dataset.
	if dataset_name in ['cqt', 'stft', 'mfcc', 'chroma']:
		tf_representation = file_manager.load(ind=0, data_type=dataset_name) # change to more general name than 'tf_represnetation'
		tf_height = HEIGHT[dataset_name]
		tf_width = int(6 * CQT_CONST["frames_per_sec"]) # 6-seconds		
	elif dataset_name in ['label']:
		pass
	else:
		print '??? dataset name wrong.'
	
	path_in = PATH_HDF + 'temp_' + dataset_name + '/' # path to read numpy files

	# create or load dataset
	if os.path.exists(PATH_HDF_TEMP + filename):
		file_write = h5py.File(PATH_HDF_TEMP + filename, 'r+')
		print 'loading hdf file that exists already there.'
	else:
		file_write = h5py.File(PATH_HDF_TEMP + filename, 'w')
		print 'creating new hdf file.'

	if dataset_name == 'label':
		for dim_label in xrange(2, 21):
			dataset_name_num = dataset_name + str(dim_label)
			if dataset_name_num in file_write:
				data_to_store = file_write[dataset_name_num]
			else:
				data_to_store = file_write.create_dataset(dataset_name_num, (num_clips, dim_label))
			labels = np.load(PATH_DATA + (FILE_DICT["mood_latent_tfidf_matrix"] % dim_latent_feature))
			for data_idx, song_idx in enumerate(song_file_inds):
				for clip_idx in clips_per_song:
					data_to_store[data_idx + clip_idx*num_songs, :] = labels[song_idx, :]
		file_write.close()
		print 'Writing labels in hdfs: done for label in range(2, 20'
		return
	else:
		if dataset_name in file_write:
			data_to_store = file_write[dataset_name]
		else:
			data_to_store = file_write.create_dataset(dataset_name, (num_clips, 1, tf_height, tf_width), 
													maxshape=(None, None, None, None)) #(num_samples, num_channel, height, width)
	
	# fill the dataset.
	done_idx_file_path = PATH_HDF_TEMP + filename + '_' +dataset_name + '_done_idx.npy'
	if os.path.exists(done_idx_file_path):
		done_idx = np.load(done_idx_file_path)
	else:
		done_idx = -1
	for dataset_idx in xrange(len(song_file_inds)):
		if dataset_idx <= done_idx:
			continue			
		song_idx = song_file_inds[dataset_idx]
		track_id = track_ids[song_idx]
		# put this cqt selection into hdf dataset.
		tf_selections = np.load(path_in + str(track_id) + '.npy')
		for clip_idx in range(clips_per_song):
			data_to_store[dataset_idx + clip_idx*num_songs, 0, :, :] =  tf_selections[:,:,clip_idx]
		print 'Done: cp2hdf, dataset_idx:%d, track_id: %d' % (dataset_idx, track_id)
		np.save(done_idx_file_path, dataset_idx)

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
	# example: $ python main_convert_cp2hdf.py b stft test
		file_manager = my_utils.File_Manager()
		train_inds, valid_inds, test_inds = file_manager.split_inds(num_folds=10)
		if sys.argv[3] == 'train':
			create_hdf_dataset(filename='data_train.h5', 
								dataset_name=datatype,
								file_manager=file_manager,
								song_file_inds=train_inds)
		elif sys.argv[3] == 'valid':
			create_hdf_dataset(filename='data_valid.h5', 
								dataset_name=datatype,
								file_manager=file_manager,
								song_file_inds=valid_inds)
		elif sys.argv[3] == 'test':
			create_hdf_dataset(filename='data_test.h5', 
								dataset_name=datatype,
								file_manager=file_manager,
								song_file_inds=test_inds)
		else:
			print '?????'
	
	
	