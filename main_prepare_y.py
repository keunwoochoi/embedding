import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""
It deals with labels (or 'y') only!
For 'data' (or 'x'), see main_prepare.y.py

It prepares stft and cqt representation.
It is recommended to rather use this file independetly than import -- because it's clearer!
"""

import platform
import os
import sys
import cPickle as cP
import numpy as np
import pdb
import time
from multiprocessing import Pool
from environments import *
from constants import *
import my_utils


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

if __name__=='__main__':
	label_type='mood' # 'mood', 'genre'
	# preproess() # read text file and generate dictionaries.
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
		mood_tags_tfidf_matrix = get_tfidf()
		for k in xrange(2,21):
			filename_out = FILE_DICT["mood_latent_tfidf_matrix"] % k
			if os.path.exists(PATH_DATA + filename_out):
				W = np.load(PATH_DATA + filename_out)
			else:
				W = get_LDA(X=mood_tags_tfidf_matrix, num_components=k, show_topics=True)
				# for ind, row in enumerate(W): # normaliseeh2nfqpfg

					# W[ind,:] = W[ind,:]/np.linalg.norm(W[ind,:])

				np.save(PATH_DATA + filename_out, W)

