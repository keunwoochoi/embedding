import platform
import os
import sys
import cPickle as cP
import numpy as np
import pdb

device_name = platform.node()

if device_name.endswith('eecs.qmul.ac.uk') or device_name in ['octave']:
	isServer = True
	isMac = False
	isMsi = False

else:
	print 'unknown device'

if isServer:
	print "THIS IS A SERVER NAMED %s" % device_name
	PATH_ILM = '/import/c4dm-01/ilm10k-dataset/'
	PATH_ILM_ACT = PATH_ILM + 'act-coordinates/'
	PATH_ILM_AUDIO = PATH_ILM + 'ilmaudio/'
	PATH_ILM_META = PATH_ILM + 'metadata/'

	PATH_HOME = "/homes/kc306/"
	PATH_WORK = PATH_HOME + "embedding/"
	PATH_DATA = PATH_WORK + "data/"

	PATH_STFT = '/import/c4dm-01/ilm10k_audio_transformed/' + 'STFT/'
	PATH_CQT  = '/import/c4dm-01/ilm10k_audio_transformed/' + 'CQT/'

	sys.path.append(PATH_HOME + 'modules/' + 'librosa/')

import librosa


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
			tags_matrix.append(line_array)

	np.save(PATH_DATA + "mood_tags_matrix.np", np.array(tags_matrix))

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

def prepare_stft():
	import librosa

	SR = 11025
	N_FFT = 1024
	WIN_LEN = 1024
	HOP_LEN = 512

	dict_id_path = cP.load(open(PATH_DATA + "id_path_dict_w_audio.cP", "r"))
	track_ids = cP.load(open(PATH_DATA + "track_ids_w_audio.cP", "r"))
	num_tracks = len(track_ids)

	for ind, track_id in enumerate(track_ids):
		src, sr = librosa.load(PATH_ILM_AUDIO + dict_id_path[track_id], sr=SR, mono=False)
		pdb.set_trace()
		SRC = librosa.stft(src, n_fft = N_FFT, hop_length=HOP_LEN, win_length = WIN_LEN)
		np.save( PATH_STFT + str(track_id) + '.npy', SRC)
		SRC_cqt = librosa.logamplitude(librosa.cqt(src, sr=SR, hop_length=HOP_LEN, bins_per_octave=24, n_bins=24*8)**2, ref_power=1.0)
		np.save( PATH_CQT + str(track_id) + '.npy', SRC_cqt)
		print "STFT and CQT for %d/%d : done" % (track_id, num_tracks)


if __name__=="__main__":
	preprocess()
	print '---preprocess: done---'
	prepare_stft()





