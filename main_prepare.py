import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
	print '---preprocess: done---'

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
	print 'will do chroma'
	CQT = 10**(0.05*CQT) # log_am --> linear (with ref_power=1.0)
	chroma_left = librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,0], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])
	chroma_right= librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,1], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])
	chroma_mono = librosa.feature.chroma_cqt(y=None, sr=CQT_CONST["sr"], C=CQT[:,:,0]+CQT[:,:,1], 
		                                     hop_length=CQT_CONST["hop_len"], 
		                                     bins_per_octave=CQT_CONST["bins_per_octave"])

	np.save(PATH_CHROMA+str(track_id)+'.npy', 
			librosa.logamplitude(np.dstack((chroma_left, chroma_right, chroma_mono))))
	print "Done: %s, chroma" % str(track_id)

def do_pitchgram(CQT, track_id):
	'''new way of representation, should be called as 
	log-harmonigram or something.
	returns a CQT that is re-ordered in frequency band.
	'''
	print 'will do pitchgram'
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
	print 'will do hps_on_cqt'
	print CQT.shape
	
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

def load_cqt(track_id):
	return np.load(PATH_CQT + str(track_id) + '.npy')

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
	''''''
	if os.path.exists(PATH_CHROMA + str(track_id) + '.npy'):
		print "chroma:skip this id: %d, it's already there!" % track_id
	else:
		CQT = load_cqt(track_id)
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
	
	if task not in ['stft', 'cqt', 'mfcc', 'chroma', 'hgram', 'pgram', 'hps_on_cqt', 'all_about_cqt']:
		print 'wrong argument, choose stft, cqt, mfcc, chroma, hgram, pgram, hps_on_cqt, all_about_cqt'
		sys.exit()
	if arguments[4] == 'test':
		prepare_transforms_detail(num_process, ind_process, task, isTest=True)
	else:
		prepare_transforms_detail(num_process, ind_process, task, isTest=False)

	print "#"*60
	print "FIN - using %d processes, %d-ind batch." % (num_process, ind_process)

	print "#"*60

def get_LSI(X, num_components=10):
	""" Latent Semantic Indexing. (equivalent of SVD for term-doc matrix.) 
	21 Nov 2015, Keunwoo Choi
	
	Here, LSI is ready for song-tag matrix, not term-doc matrix.
	Synonym problem is expected to be attacked effectively!
	
	First, it loads song-tag matrix, of which size is 9320-by-100 at the moment.
	Then it reduces the matrix #song-by-#tag into #song-by-k, where k<100 and a reasonablly small
	number to represent the semantic meaning of songs in tag space.
	It utilise sklearn.decomposition.TruncatedSVD
	"""
	from sklearn.decomposition import TruncatedSVD
	if X == None:
		print 'X is omitted, so just assume it is the mood tag mtx w audio.'
		X = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100

	svd = TruncatedSVD(n_components=num_components, random_state=42, n_iter=10)
	svd.fit(X) # train with given matrix. #actually, fit_transform is faster.
	reduced_matrix = svd.transform(X) # 9320-by-k(10)

	recovered_matrix = svd.inverse_transform(reduced_matrix)
	average_error = np.sqrt(np.sum((X - recovered_matrix))**2)/(X.shape[0]*X.shape[1])
	print "SVD done with k=%d, average error:%2.4f" % (num_components, average_error)

	return reduced_matrix

def get_LDA(X, num_components=10, show_topics=True):
	""" Latent Dirichlet Allication by NMF.
	21 Nov 2015, Keunwoo Choi

	LDA for a song-tag matrix. The motivation is same as get_LSI. 
	With NMF, it is easier to explain what each topic represent - by inspecting 'H' matrix,
	where X ~= X' = W*H as a result of NMF. 
	It is also good to have non-negative elements, straight-forward for both W and H.

	"""

	from sklearn.decomposition import NMF
	if X == None:
		print 'X is omitted, so just assume it is the mood tag mtx w audio.'
		X = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100

	nmf = NMF(init='nndsvd', n_components=num_components, max_iter=400) # 400 is too large, but it doesn't hurt.
	W = nmf.fit_transform(X)
	H = nmf.components_
	print '='*60
	print "NMF done with k=%d, average error:%2.4f" % (num_components, nmf.reconstruction_err_/(X.shape[0]*X.shape[1]))

	term_rankings = []
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	for topic_index in range( H.shape[0] ):
		top_indices = np.argsort( H[topic_index,:] )[::-1][0:10]
		term_ranking = [moodnames[i] for i in top_indices]
		term_rankings.append(term_ranking)
		if show_topics:	
			print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )
	print '='*60
	cP.dump(term_rankings, open(PATH_DATA + (FILE_DICT["mood_topics_strings"] % num_components), 'w'))
	return W / np.max(W) # return normalised matrix, [0, 1]

def get_tfidf():
	"""Compute tf-idf weighted matrix for song-moodtag matrix """
	if os.path.exists(PATH_DATA + FILE_DICT["mood_tags_tfidf_matrix"]):
		print 'get_tfidf() returns pre-computed tf-idf matrix'
		return np.load(PATH_DATA + FILE_DICT["mood_tags_tfidf_matrix"])

	mood_tags_matrix = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) # linear tf value.. # 9380-by-100
	
	mood_tags_matrix = np.log(1 + mood_tags_matrix) # log-weigted tf

	N_songs, N_tags = mood_tags_matrix.shape
	N_documents_contains_tags = np.zeros((1, N_tags))

	for tag_ind in xrange(N_tags):
		N_documents_contains_tags[0, tag_ind] = np.count_nonzero(mood_tags_matrix[:, tag_ind])

	idf = np.log(1 + N_songs / N_documents_contains_tags) # 1-by-N_tags idf for each.
	idf_matrix = np.tile(idf, (N_songs,1)) # idf matrix

	mood_tags_tfidf_matrix = np.multiply(mood_tags_matrix, idf_matrix)
	max_val = np.max(mood_tags_tfidf_matrix)
	if max_val != 0:
		mood_tags_tfidf_matrix = mood_tags_tfidf_matrix / max_val
	np.save(PATH_DATA + FILE_DICT["mood_tags_tfidf_matrix"], mood_tags_tfidf_matrix)

	return mood_tags_tfidf_matrix


if __name__=="__main__":

	# preprocess() # read text file and generate dictionaries.
	
	if False or "if in a case I'd like to convert more songs or other transformations ":
		prepare_transforms(sys.argv)

	# tf-idf
	print '## tf-idf?'
	if False and "it is done.":
		mood_tags_tfidf_matrix = get_tfidf()

	# [0] analysis.
	# mood_tags_matrix = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100
	print '## LSI?'
	if False and "it is already done.":
		for k in [2,3,5,10,20]:
	 		get_LSI(X=mood_tags_matrix, num_components=k)

	print '## LSI???'
	# [1] analysis - LSI
	if False and "it is already done.":
		for k in [2,3,5,10,20]:
			filename_out = FILE_DICT["mood_latent_matrix"] % k
			if os.path.exists(PATH_DATA + filename_out):
				W = np.load(PATH_DATA + filename_out)
			else:
				W = get_LSI(X=mood_tags_matrix, num_components=k, show_topics=True)
				np.save(PATH_DATA + filename_out, W)
	
	print '## LDA?'
	# analysis - LDA 
	if False and "it is already done.":
		for k in xrange(2,21):
			filename_out = FILE_DICT["mood_latent_tfidf_matrix"] % k
			if os.path.exists(PATH_DATA + filename_out):
				W = np.load(PATH_DATA + filename_out)
			else:
				W = get_LDA(X=mood_tags_tfidf_matrix, num_components=k, show_topics=True)
				for ind, row in enumerate(W):
					W[ind,:] = W[ind,:]/np.linalg.norm(W[ind,:])

				np.save(PATH_DATA + filename_out, W)
	print '## structure segmentation?'
	sys.exit(0)
	# structural segmentation
	if False or 'after understand input arguments of msaf':
		print 'start using msaf'
		import msaf
		track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
		dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))

		start = time.clock()
		for track_id in track_ids[0:10]:
			print '...for ' + PATH_ILM_AUDIO + dict_id_path[track_id]
			boundaries, labels = msaf.process(PATH_ILM_AUDIO + dict_id_path[track_id], boundaries_id="cnmf", labels_id="cnmf")
			print 'msaf: cnmf done'
		until = time.clock()
		time_cnmf = until - start
		start = time.clock()
		for track_id in track_ids[0:10]:
			boundaries, labels = msaf.process(PATH_ILM_AUDIO + dict_id_path[track_id], boundaries_id="scluster", labels_id="scluster")
			print 'msaf: scluster done'
		until = time.clock()
		time_scluster = until - start
		print "time comsumed : %f vs %f" % (time_cnmf, time_scluster)




