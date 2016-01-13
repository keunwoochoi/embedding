"""functions and modules that are used in main_prepare.
25 Dec 2015, Keunwoo Choi"""

import os
import sys
import cPickle as cP
import numpy as np
import cPickle as cP
import msaf
import my_utils
from multiprocessing import Pool

from environments import *
from constants import *
import time
import pdb

def process_boundaries(path_to_read):
	try:
		boundaries, labels = msaf.process(path_to_read, n_jobs=1,
										boundaries_id="scluster", 
										labels_id="scluster")
	except ValueError:
		boundaries = []
		labels = []
		print 'Perhaps path error:%s' % path_to_read	
	print 'boundary and label: done: %s' % path_to_read
	return (boundaries, labels)
	
def get_boundaries_all(isTest=False):
	"""get boundaries and labels using msaf. """
	if os.path.exists(PATH_DATA + FILE_DICT["segmentation"]):
		print 'Boundary file already exists: %s' % (PATH_DATA + FILE_DICT["segmentation"])
		print 'Please remove the file first to proceed.'
		return
	start = time.time()
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))
	
	if isTest:
		track_ids = track_ids[0:3]
		dict_id_path_small = {}
		[dict_id_path_small.update({track_id:dict_id_path[track_id]}) for track_id in track_ids]
		dict_id_path = dict_id_path_small

	paths_to_pass = []
	[paths_to_pass.append(PATH_ILM_AUDIO + dict_id_path[track_id]) for track_id in track_ids]
	
	print 'msaf for %d songs:' % len(paths_to_pass)
	
	ret = {}
	if True:
		#nested multiprocessing doesn't work for msaf
		p = Pool(24)
		results = p.map(process_boundaries, paths_to_pass)
		p.close()
		p.join()
		for ind, track_id in enumerate(track_ids):
			ret[track_id] = results[ind]
	else:
		for idx_path, path in enumerate(paths_to_pass):
			ret[track_ids[idx_path]] = process_boundaries(path)
			
	time_consumed = time.time() - start
	print 'boundary and labelling done! - for %d seconds' % time_consumed

	cP.dump(ret, open(PATH_DATA + FILE_DICT["segmentation"], "w"))

	return


def postprocess_boundaries():
	'''load segmentation dictionary, process labels so that
	- consider only segmentation longer than 6-s
	- same label --> merge??????? (not sure)
	- compute average energy of each segment,

	*** NOT all the results has 3> segments. Even 0 segment exists. 
	Perhaps a workaround should be employed. 

	'''
	file_manager = my_utils.File_Manager()
	if os.path.exists(PATH_DATA + FILE_DICT["segment_selection"]):

		segment_selection = cP.load(open(PATH_DATA + FILE_DICT["segment_selection"], 'r')) # dictionary of key and list, which is consists of tuples (frame_strt, frame_end) for segments
		begin_idx = len(segment_selection)
		print 'Load from previously processed segment selection, which is done before %d' % begin_idx
	else:
		segment_selection = {}
		begin_idx = 0
		
	dict_segmentation = cP.load(open(PATH_DATA + FILE_DICT["segmentation"], 'r')) # track_id : (boundaries, labels)
	#track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	frame_per_sec = SR / HOP_LEN
	
	min_num_selected = 999999
	for idx, track_id in enumerate(file_manager.track_ids):
		if idx < begin_idx:
			continue
		# load cqt
		CQT = 10**(0.05*file_manager.load_cqt(idx))
		CQT = CQT ** 2 # energy.
		CQT = np.sum(CQT, axis=2) # downmix
		frame_energies = np.sum(CQT, axis=0) # sum of energy in each frame
		# load boundaries
		boundaries, labels = dict_segmentation[track_id]
		if len(boundaries) < 1:
			boundaries = np.array([0, len(frame_energies)])
			labels = np.array([1])
		if isinstance(boundaries, list):
			boundaries = np.array(boundaries)
			labels = np.array(labels)
		# compute mean energy, only for segments >= 6-seconds
		boundaries = np.round(frame_per_sec*boundaries).astype(np.int32) # [sec] --> [frame]
		boundaries[0] = 0
		boundaries[-1] = len(frame_energies) 
		average_energies = []
		long_average_energies = []
		long_boundaries = []
		long_labels = []
		for b_idx, b_from in enumerate(boundaries[:-1]):
			b_to = boundaries[b_idx+1]
			average_energies.append(np.mean(frame_energies[b_from:b_to]))
			if b_to - b_from <= frame_per_sec*6:
				continue
			long_boundaries.append((b_from, b_to))
			long_average_energies.append(np.mean(frame_energies[b_from:b_to]))
			long_labels.append(labels[b_idx])
		# pick segments.
		order = np.argsort(long_average_energies) # increasing order
		order = order[::-1] # decreasing order
		result = []
		labels_added = []
		for segment_idx in order:
			if long_labels[segment_idx] not in labels_added:
				result.append(long_boundaries[segment_idx])
				labels_added.append(long_labels[segment_idx])
		segment_selection[track_id] = result
		print 'idx %d, track_id %d : Done for boundary post processing, %d segments selected.' % (idx, track_id, len(result))
		if idx % 300 == 0:
			print '...saving...'
			cP.dump(segment_selection, open(PATH_DATA + FILE_DICT["segment_selection"], 'w')) # dictionary of key and list, which is consists of tuples (frame_strt, frame_end) for segments

	cP.dump(segment_selection, open(PATH_DATA + FILE_DICT["segment_selection"], 'w')) # dictionary of key and list, which is consists of tuples (frame_strt, frame_end) for segments
	print "DONE:preprocessing of selecting clips from songs using msaf"
	return
