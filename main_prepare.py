import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
It deals with 'data' (or 'x') only!
For labels (or 'y'), see main_prepare.y.py

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
import my_utils
import preprocessors


def do_mfcc(src, track_id):
	'''src would be stereo for ilm10k.'''
	def augment_mfcc(mfcc):
		'''concatenate d-mfcc and dd-mfcc.
		mfcc: numpy 2d array.'''
		def get_derivative_mfcc(mfcc):
			'''return a same-sized, derivative of mfcc.'''
			len_freq, num_fr = mfcc.shape
			mfcc = np.hstack((np.zeros((len_freq, 1)), mfcc))
			return mfcc[:, 1:] - mfcc[:, :-1]
		d_mfcc = get_derivative_mfcc(mfcc)
		return np.vstack((mfcc, d_mfcc, get_derivative_mfcc(d_mfcc)))

	mfcc_left = librosa.feature.mfcc(src[0,:], sr=SR, n_mfcc=20)
	mfcc_right= librosa.feature.mfcc(src[1,:], sr=SR, n_mfcc=20)
	mfcc_mono = librosa.feature.mfcc(0.5*(src[0,:]+src[1,:]), sr=SR, n_mfcc=20)
	mfcc_left = mfcc_left[1:, :] # remove the first coeff.
	mfcc_right= mfcc_right[1:, :]
	mfcc_mono = mfcc_mono[1:, :]
	mfcc_left = augment_mfcc(mfcc_left)
	mfcc_right= augment_mfcc(mfcc_right)
	mfcc_mono = augment_mfcc(mfcc_mono)

	np.save(PATH_MFCC + str(track_id) + '.npy', np.dstack((mfcc_left, mfcc_right, mfcc_mono)))
	print "Done: %s, mfcc" % str(track_id)

def do_stft(src, track_id):
	SRC_L = librosa.stft(src[0,:], n_fft = N_FFT, hop_length=HOP_LEN, win_length = WIN_LEN)
	SRC_R = librosa.stft(src[1,:], n_fft = N_FFT, hop_length=HOP_LEN, win_length = WIN_LEN)
	np.save(PATH_STFT + str(track_id) + '.npy', np.dstack((SRC_L, SRC_R)))
	print "Done: %s" % str(track_id)

def do_melgram(src, track_id):
	SRC = librosa.feature.melspectrogram(y=src[0,:]+src[1,:], n_fft=N_FFT, hop_length=HOP_LEN, n_mels=128) # MONO!
	np.save(PATH_MELGRAM + str(track_id) + '.npy', np.dstack((SRC)))
	print "Done: %s" % str(track_id)	

def do_cqt(src, track_id):
	SRC_cqt_L = librosa.logamplitude(librosa.cqt(src[0,:], sr=CQT_CONST["sr"], 
									 hop_length=CQT_CONST["hop_len"], 
		                             bins_per_octave=CQT_CONST["bins_per_octave"], 
		                             n_bins=CQT_CONST["n_bins"])**2, ref_power=1.0)
	SRC_cqt_R = librosa.logamplitude(librosa.cqt(src[1,:], sr=CQT_CONST["sr"], 
									 hop_length=CQT_CONST["hop_len"], 
		                             bins_per_octave=CQT_CONST["bins_per_octave"], 
		                             n_bins=CQT_CONST["n_bins"])**2, ref_power=1.0)
	np.save(PATH_CQT + str(track_id) + '.npy', np.dstack((SRC_cqt_L, SRC_cqt_R)))
	print "Done: %s" % str(track_id)

def do_chroma_cqt(CQT, track_id):
	'''compute chroma feature from CQT representation (stereo)
	unlike other 'do' methods, this load uses CQT.

	input CQT: log-amplitude.
	'''
	
	CQT = 10**(0.05*CQT) # log_am --> linear (with ref_power=1.0)
	chroma_left = librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,0], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     n_chroma=CQT_CONST["bins_per_octave"],
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])
	chroma_right= librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,1], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     n_chroma=CQT_CONST["bins_per_octave"],
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])
	chroma_mono = librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,0]+CQT[:,:,1], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     n_chroma=CQT_CONST["bins_per_octave"],
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])

	np.save(PATH_CHROMA+str(track_id)+'.npy', 
			librosa.logamplitude(np.dstack((chroma_left, chroma_right, chroma_mono))))
	print "Done: %s, chroma" % str(track_id)

def do_pitchgram(CQT, track_id):
	'''new way of representation, should be called as 
	log-harmonigram or something.
	returns a CQT that is re-ordered in frequency band.
	'''
	
	ret = np.zeros(CQT.shape)
	for depth_cqt in xrange(CQT.shape[2]):
		for octave in xrange(CQT_CONST["num_octaves"]):
			for bin in xrange(CQT_CONST["bins_per_octave"]):
				cqt_bin_idx = octave*CQT_CONST["bins_per_octave"]+bin
				ret_bin_idx = bin*CQT_CONST["num_octaves"] + octave
				ret[ret_bin_idx, :, depth_cqt] = CQT[cqt_bin_idx, :, depth_cqt]
	np.save(PATH_PGRAM+str(track_id)+'.npy', ret)
	print "Done: %s, Pitchgram - pitch class collection on cqt" % str(track_id)

def do_HPS_on_CQT(CQT, track_id):
	'''HPS on CQT
		input CQT: log-amplitude.
	'''
	
	CQT = 10**(0.05*CQT) # log_am --> linear (with ref_power=1.0)
	ret_H = np.zeros(CQT.shape)
	ret_P = np.zeros(CQT.shape)
	for depth_cqt in xrange(CQT.shape[2]):
		ret_H[:,:,depth_cqt], ret_P[:,:,depth_cqt] = librosa.decompose.hpss(CQT[:,:,depth_cqt])
	np.save(PATH_CQT_H+str(track_id)+'.npy', librosa.logamplitude(ret_H))
	np.save(PATH_CQT_P+str(track_id)+'.npy', librosa.logamplitude(ret_P))
	print "Done: %d, HPS for CQT " % track_id

def do_harmonigram(STFT, track_id, sr, n_fft):
	'''
	harmonigram 
	STFT.shape = (n_fft/2+1, num_frame, 3) # for left, right, and mono.
	not sure if I really need to use it. 
	'''
	f_min = 110
	f_max = 880
	f_gap = float(sr) / n_fft
	idx_min = int(np.ceil(f_min/f_gap) + 1)
	idx_max = int(np.ceil(f_max/f_gap))
	num_ret_bin = idx_max - idx_min + 1
	ret_shape = STFT.shape
	ret_shape[0] = num_ret_bin 
	ret = np.zeros(ret_shape)
	
	for ret_idx in range(num_ret_bin):
		stft_idx = ret_idx + idx_min
		gap_stft_idx = stft_idx - 1
		
		for count in range(11): #compute to 10-th harmonic. 
			ret[ret_idx, :, :] += STFT[stft_idx, :, :]
			stft_idx += gap_stft_idx
	np.save(PATH_HGRAM+str(track_id)+'.npy', ret)
	print "Done: %s, Harmonigram - pitch class collection on cqt" % str(track_id)

def load_src(track_id):
	dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))
	src, sr = librosa.load(PATH_ILM_AUDIO + dict_id_path[track_id], sr=SR, mono=False)
	return src, sr

def load_cqt(track_id, option=None):
	if option is None:
		return np.load(PATH_CQT + str(track_id) + '.npy')
	elif option == 'h':
		return np.load(PATH_CQT_H + str(track_id) + '.npy')
	else:
		raise RuntimeError('wrong option for load_cqt')

def load_stft(track_id):
	return np.load(PATH_STFT + str(track_id) + '.npy')

def process_stft(track_id):
	if os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		print "stft: skip this id: %d, it's already there!" % track_id
	else:
		src, sr = load_src(track_id)
		do_stft(src, track_id)

def process_cqt(track_id):
	if os.path.exists(PATH_CQT + str(track_id) + '.npy'):
		print "cqt :skip this id: %d, it's already there!" % track_id
	else:
		src, sr = load_src(track_id)
		do_cqt(src, track_id)

def process_hps_on_cqt(track_id):
	if os.path.exists(PATH_CQT_H + str(track_id) + '.npy') and os.path.exists(PATH_CQT_P + str(track_id) + '.npy'):
		print "hps on cqt :skip this id: %d, it's already there!" % track_id
	else:
		CQT = load_cqt(track_id)
		do_HPS_on_CQT(CQT, track_id)

def process_mfcc(track_id):
	if os.path.exists(PATH_MFCC + str(track_id) + '.npy'):
		print "mfcc:skip this id: %d, it's already there!" % track_id
	else:
		src, sr = load_src(track_id)
		do_mfcc(src, track_id)	


def process_chroma(track_id):
	'''
	option: none: do chroma based on normal CQT
			h: based on harmonic.

	'''
	if os.path.exists(PATH_CHROMA + str(track_id) + '.npy'):
		print "chroma:skip this id: %d, it's already there!" % track_id
	else:
		CQT = load_cqt(track_id, option='h')
		do_chroma_cqt(CQT, track_id)	


def process_pitchgram(track_id):
	if os.path.exists(PATH_PGRAM + str(track_id) + '.npy'):
		print "pgram:skip this id: %d, it's already there!" % track_id
	else:
		CQT = load_cqt(track_id)
		do_pitchgram(CQT, track_id)
		

def process_harmonigram(track_id):
	if os.path.exists(PATH_HGRAM + str(track_id) + '.npy'):
		print "hgram:skip this id: %d, it's already there!" % track_id
	else:
		STFT = load_stft(track_id)
		do_harmonigram(STFT, track_id, SR, N_FFT)

def process_all_about_cqt(track_id):
	'''do hps_on_cqt, chroma, pitchgram'''
	if not os.path.exists(PATH_CQT + str(track_id) + '.npy'):
		print 'cqt missing: will do: %d' % track_id
		process_cqt(track_id)

	if os.path.exists(PATH_CQT_H + str(track_id) + '.npy'):
		if os.path.exists(PATH_CQT_P + str(track_id) + '.npy'):
			if os.path.exists(PATH_CHROMA + str(track_id) + '.npy'):
				if os.path.exists(PATH_PGRAM + str(track_id) + '.npy'):
					print "skip: %d" % track_id
					return
	
	try:
		CQT = load_cqt(track_id)
	except ValueError:
		print 'value error on: %d. please remove the cqt and do this again.' % track_id
	if not ( os.path.exists(PATH_CQT_H + str(track_id) + '.npy') and os.path.exists(PATH_CQT_P + str(track_id) + '.npy')  ):
		do_HPS_on_CQT(CQT, track_id)
	if not os.path.exists(PATH_CHROMA + str(track_id) + '.npy'):
		do_chroma_cqt(CQT, track_id)	
	if not os.path.exists(PATH_PGRAM + str(track_id) + '.npy'):
		do_pitchgram(CQT, track_id)
	print "Done: %d all about cqt" % track_id

def process_stft_cqt(track_id):
	if os.path.exists(PATH_CQT + str(track_id) + '.npy') and os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		print "stft & cqt: skip this id: %d, it's already there!" % track_id
	elif os.path.exists(PATH_CQT + str(track_id) + '.npy'):
		src, sr = load_src(track_id)
		do_stft(src, track_id)
	elif os.path.exists(PATH_STFT + str(track_id) + '.npy'):
		src, sr = load_src(track_id)
		do_cqt(src, track_id)
	else:
		src, sr = load_src(track_id)
		do_cqt(src, track_id)
		do_stft(src, track_id)

def process_melgram(track_id):
	if os.path.exists(PATH_MELGRAM + str(track_id) + '.npy'):
		# if os.path.getsize(PATH_MELGRAM + str(track_id) + '.npy') != 0: # have no permission
		print "melgram: skip this id: %d, it's already there!" % track_id
	
	src, sr = load_src(track_id)
	do_melgram(src, track_id)

def prepare_transforms_detail(num_process, ind_process, task, isTest):

	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	num_tracks = len(track_ids)
	num_of_server_used = 6
	num_subsets = num_tracks/num_of_server_used
	print "prepare_transforms_detail(); dictionaries loaded"
	
	if ind_process == -1:
		print "ind_process == -1, so will do all!"
		track_ids_here = track_ids
	elif ind_process == num_of_server_used-1: 
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
		p.map(process_stft, track_ids_here)
	elif task == 'cqt':
		p.map(process_cqt, track_ids_here)
	elif task == 'stft_cqt':
		p.map(process_stft_cqt, track_ids_here)
	elif task == 'mfcc':
		p.map(process_mfcc, track_ids_here)
	elif task == 'chroma':
		p.map(process_chroma, track_ids_here)	
	elif task == 'hgram':
		p.map(process_harmonigram, track_ids_here)	
	elif task=='pgram':
		p.map(process_pitchgram, track_ids_here)
	elif task=='hps_on_cqt':
		p.map(process_hps_on_cqt, track_ids_here)
	elif task=='all_about_cqt':
		p.map(process_all_about_cqt, track_ids_here)
	elif task=='melgram':
		p.map(process_melgram, track_ids_here)
	else:
		print 'task name undefined: %s' % task
		pass
	
	p.close()
	p.join()
	print "total time: %0.2f seconds" % (time.time()-start)
	print "average %0.2f seconds per song" % ((time.time()-start)/len(track_ids_here))

def prepare_transforms(arguments):
	"""Multiprocessing-based stft or cqt conversion for all audio files. 
	"""
	def print_usage():
		print "filename number_core, [number_index](0-5), [STFT or CQT] [test or real]."
		print "number of index is based on 0"
		print "This enables to run this code over many school servers."
		print "ex) $ python main_prepare.py 12 0 cqt real     in server 1," 
		print "ex) $ python main_prepare.py 12 1 cqt real     in server 2 "
		print "ex) $ python main_prepare.py 12 -1 cqt real    --> with -1, it will do the whole set."

	if len(arguments) < 5:
		print_usage()
		sys.exit()
	num_process = int(arguments[1])
	ind_process = int(arguments[2])
	task = arguments[3].lower()
	print num_process, " processes"
	
	if task not in ['stft', 'cqt', 'mfcc', 'chroma', 'hgram', 'pgram', 'hps_on_cqt', 'all_about_cqt', 'melgram']:
		print 'wrong argument, choose stft, cqt, mfcc, chroma, hgram, pgram, hps_on_cqt, all_about_cqt'
		sys.exit()
	if arguments[4] == 'test':
		prepare_transforms_detail(num_process, ind_process, task, isTest=True)
	else:
		prepare_transforms_detail(num_process, ind_process, task, isTest=False)

	print "#"*60
	print "FIN - using %d processes, %d-ind batch." % (num_process, ind_process)

	print "#"*60

if __name__=="__main__":
	
	if False or "if in a case I'd like to convert more songs or other transformations ":
		prepare_transforms(sys.argv)
		sys.exit(0)

	
	print '## structure segmentation. add any argument to test it.'
	# structural segmentation
	if False or 'after understand input arguments of msaf':
		
		if len(sys.argv) == 1:
			print 'msaf for the whole song!'
			preprocessors.get_boundaries_all(isTest=False)
		else:
			preprocessors.get_boundaries_all(isTest=True)
	# pick the most important segments
	if False or 'after segmentation, select segments.':
		preprocessors.postprocess_boundaries()
		


