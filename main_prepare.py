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

def prepare_transforms(arguments):
	"""Multiprocessing-based stft or cqt conversion for all audio files. 
	"""
	def print_usage():
		print "filename number_core, [number_index], [STFT or CQT] [test or real]."
		print "number of index is based on 0"
		print "This enables to run this code over many school servers."

	if len(arguments) < 5:
		print_usage()
		sys.exit()
	num_process = int(arguments[1])
	ind_process = int(arguments[2])
	task = arguments[3].lower()
	print num_process, " processes"
	
	if task not in ['stft', 'cqt']:
		print 'wrong argument, choose stft or cqt'
		sys.exit()
	if arguments[4] == 'test':
		prepare_stft(num_process, ind_process, task, isTest=True)
	else:
		prepare_stft(num_process, ind_process, task, isTest=False)

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

	nmf = NMF(init='nndsvd', n_components=num_components, max_iter=200)
	W = nmf.fit_transform(X)
	H = nmf.components_
	print '='*60
	print "NMF done with k=%d, average error:%2.4f" % (num_components, nmf.reconstruction_err_/(X.shape[0]*X.shape[1]))

	if show_topics:

		moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
		for topic_index in range( H.shape[0] ):
			top_indices = np.argsort( H[topic_index,:] )[::-1][0:10]
			term_ranking = [moodnames[i] for i in top_indices]
			print "Topic %d: %s" % ( topic_index, ", ".join( term_ranking ) )
		print '='*60
	return W / np.max(W) # return normalised matrix, [0, 1]

def get_tfidf():
	"""Compute tf-idf weighted matrix for song-moodtag matrix """
	if os.path.exists(PATH_DATA + FILE_DICT["mood_tags_tfidf_matrix"]):
		print 'get_tfidf() returns pre-computed tf-idf matrix'
		return np.load(PATH_DATA + FILE_DICT["mood_tags_tfidf_matrix"], mood_tags_tfidf_matrix)

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
	
	# prepare_transforms(sys.argv)

	# tf-idf
	# mood_tags_tfidf_matrix = get_tfidf()

	# [0] analysis.
	# mood_tags_matrix = np.load(PATH_DATA + FILE_DICT["mood_tags_matrix"]) #np matrix, 9320-by-100
	if False and "it is already done.":
		for k in [2, 3, 5, 10, 20]:
	 		get_LSI(X=mood_tags_matrix, num_components=k)

	# [1] analysis - LSI
	if False and "it is already done.":
		for k in [2,3,5,10,20]:
			filename_out = FILE_DICT["mood_latent_matrix"] % k
			if os.path.exists(PATH_DATA + filename_out):
				W = np.load(PATH_DATA + filename_out)
			else:
				W = get_LDA(X=mood_tags_matrix, num_components=k, show_topics=True)
				np.save(PATH_DATA + filename_out, W)
	# analysis - LDA 
	if False and "it is already done.":
		for k in [2,3,5,10,20]:
			filename_out = FILE_DICT["mood_latent_tfidf_matrix"] % k
			if os.path.exists(PATH_DATA + filename_out):
				W = np.load(PATH_DATA + filename_out)
			else:
				W = get_LDA(X=mood_tags_tfidf_matrix, num_components=k, show_topics=True)
				np.save(PATH_DATA + filename_out, W)

	# structural segmentation
	import msaf
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))
	start = time.clock()
	boundaries, labels = msaf.process(PATH_ILM_AUDIO + dict_id_path[track_id], boundaries_id="cnmf", labels_id="cnmf")
	until = time.clock()
	time_cnmf = until - start
	start = time.clock()
	boundaries, labels = msaf.process(PATH_ILM_AUDIO + dict_id_path[track_id], boundaries_id="scluster", labels_id="cnmf")
	until = time.clock()
	time_scluster = until - start
	print "time comsumed : %f vs %f" % (time_cnmf, time_scluster)


