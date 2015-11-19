"""
It prepares stft and cqt representation.
It is recommended to rather use this file independetly than import -- because it's clearer!
"""

import platform
import os
import sys
import cPickle as cP
import numpy as np
import pdb
import librosa
import time
from multiprocessing import Pool
from environments import *
from constants import *

# [0] Pre-processing: make my own dictionary.
def preprocess():
	dict_id_path = {}
	with open(PATH_ILM_META + r"collection10000-810(500)-5(1)-loglikgmm-wav.csv", 'r') as id_path_fp:
		line = id_path_fp.readline() # ignore the first line
		for line in id_path_fp: # ignore the first line
			line = line.rstrip("\n").strip('"')
			song_id, song_path = line.split('","') #	line:"444","e/audio/x-wav/eb/1508-04.01.wav"
			dict_id_path[int(song_id)] = song_path.split("/")[-1]

	cP.dump(dict_id_path, open(PATH_DATA + "id_path_dict.cP" , "w"))

	moodnames = []
	with open(PATH_ILM_ACT + "moodnames.txt", 'r') as moodnames_fp:
		for line in moodnames_fp:
			moodnames.append(line.rstrip("\n"))

	cP.dump(moodnames, open(PATH_DATA + "moodnames.cP", "w"))

	track_ids = []
	with open(PATH_ILM_ACT + "trackids.txt", 'r') as trackids_fp:
		for line in trackids_fp:
			track_ids.append(int(line.rstrip("\n")))

	cP.dump(track_ids, open(PATH_DATA + "track_ids.cP", "w"))

	tags_matrix = []
	with open(PATH_ILM_ACT + "moodtags.txt", 'r') as tags_fp:
		for line in tags_fp:
			line_array = line.rstrip("\n").split(" ")
			new_line = []
			for element in line_array:
				new_line.append(int(element))
			tags_matrix.append(new_line)

	np.save(PATH_DATA + "mood_tags_matrix.npy", np.array(tags_matrix))

	audio_exists = []
	for ind, track_id in enumerate(track_ids):
		if track_id in dict_id_path:
			audio_exists.append(True)
		else:
			audio_exists.append(False)

	track_id_w_audio = []
	tags_matrix_w_audio = []
	dict_id_path_w_audio = {}
	for ind, boolean in enumerate(audio_exists):
		if boolean:
			track_id_w_audio.append(track_ids[ind])
			tags_matrix_w_audio.append(tags_matrix[ind])
			dict_id_path_w_audio[track_ids[ind]] = dict_id_path[track_ids[ind]]

	cP.dump(dict_id_path_w_audio, open(PATH_DATA + "id_path_dict_w_audio.cP" , "w"))
	cP.dump(track_id_w_audio, open(PATH_DATA + "track_ids_w_audio.cP", "w"))
	np.save(PATH_DATA + "mood_tags_matrix_w_audio", np.array(tags_matrix_w_audio))
	'''int, int, string(stft or cqt)'''

def do_stft(src, track_id):
	SRC_L = librosa.stft(src[0,:], n_fft = N_FFT, hop_length=HOP_LEN, win_length = WIN_LEN)
	SRC_R = librosa.stft(src[1,:], n_fft = N_FFT, hop_length=HOP_LEN, win_length = WIN_LEN)
	np.save( PATH_STFT + str(track_id) + '.npy', np.dstack((SRC_L, SRC_R)))
	print "Done: %s" % str(track_id)

def do_cqt(src, track_id):
	SRC_cqt_L = librosa.logamplitude(librosa.cqt(src[0,:], sr=SR, hop_length=HOP_LEN, bins_per_octave=24, n_bins=24*7)**2, ref_power=1.0)
	SRC_cqt_R = librosa.logamplitude(librosa.cqt(src[1,:], sr=SR, hop_length=HOP_LEN, bins_per_octave=24, n_bins=24*7)**2, ref_power=1.0)
	np.save( PATH_CQT + str(track_id) + '.npy', np.dstack((SRC_cqt_L, SRC_cqt_R)) )
	print "Done: %s" % str(track_id)

def do_load(track_id):
	dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))
	src, sr = librosa.load(PATH_ILM_AUDIO + dict_id_path[track_id], sr=SR, mono=False)
	return src, sr

def do_load_stft(track_id):
	if os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		print "stft: skip this id: %d, it's already there!" % track_id
	else:
		src, sr = do_load(track_id)
		do_stft(src, track_id)

def do_load_cqt(track_id):
	if os.path.exists(PATH_CQT + str(track_id) + '.npy'):
		print "cqt :skip this id: %d, it's already there!" % track_id
	else:
		src, sr = do_load(track_id)
		do_cqt(src, track_id)

def do_load_stft_cqt(track_id):
	if os.path.exists(PATH_CQT + str(track_id) + '.npy') and os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		print "stft & cqt: skip this id: %d, it's already there!" % track_id
	elif os.path.exists(PATH_CQT + str(track_id) + '.npy'):
		src, sr = do_load(track_id)
		do_stft(src, track_id)
	elif os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		src, sr = do_load(track_id)
		do_cqt(src, track_id)
	else:
		src, sr = do_load(track_id)
		do_cqt(src, track_id)
		do_stft(src, track_id)

def prepare_stft(num_process, ind_process, task, isTest):

	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	num_tracks = len(track_ids)
	num_subsets = num_tracks/8 # because there are 8 servers I can use.
	print "prepare stft; dictionaries loaded"
	
	if ind_process == -1:
		print "ind_process == -1, so will do all!"
		track_ids_here = track_ids
	elif ind_process == 8:
		track_ids_here = track_ids[ind_process*num_subsets : ]
	else:
		track_ids_here = track_ids[ind_process*num_subsets : (ind_process+1)*num_subsets]
	
	if isTest:
		rand_ind = np.random.randint(len(track_ids_here)-2)
		track_ids_here = track_ids[rand_ind:rand_ind+2]

	print "Only %d files will be converted by task named: %s " % (len(track_ids_here), task)
	start = time.time()

	p = Pool(num_process)
	if task == 'stft':
		p.map(do_load_stft, track_ids_here)
	elif task == 'cqt':
		p.map(do_load_cqt, track_ids_here)
	elif task == 'stft_cqt':
		p.map(do_load_stft_cqt, track_ids_here)
	else:
		pass
	
	p.close()
	p.join()
	print "total time: %0.2f seconds" % (time.time()-start)
	print "average %0.2f seconds per song" % ((time.time()-start)/len(track_ids_here))
	
def print_usage():
	print "filename number_core, [number_index], [STFT or CQT] [test or real]."
	print "number of index is based on 0"

if __name__=="__main__":
	preprocess()
	# print '---preprocess: done---'


	sys.exit()
	

	if len(sys.argv) < 5:
		print_usage()
		sys.exit()
	num_process = int(sys.argv[1])
	ind_process = int(sys.argv[2])
	task = sys.argv[3].lower()
	print num_process, " processes"
	
	if task not in ['stft', 'cqt']:
		print 'wrong argument, choose stft or cqt'
		sys.exit()
	if sys.argv[4] == 'test':
		prepare_stft(num_process, ind_process, task, isTest=True)
	else:
		prepare_stft(num_process, ind_process, task, isTest=False)

	print "#"*60
	print "FIN - using %d processes, %d-ind batch." % (num_process, ind_process)

	print "#"*60



