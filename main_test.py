import platform
import os
import sys
import cPickle as cP
import numpy as np
import pdb

device_name = platform.node()

if device_name.endswith('eecs.qmul.ac.uk'):
	isServer = True
	isMac = False
	isMsi = False

else:
	print 'unknown device'

if isServer:
	PATH_ILM = '/import/c4dm-01/ilm10k-dataset/'
	PATH_ILM_ACT = PATH_ILM + 'act-coordinates/'
	PATH_ILM_AUDIO = PATH_ILM + 'ilmaudio/'
	PATH_ILM_META = PATH_ILM + 'metadata/'

	PATH_HOME = "/homes/kc306/"
	PATH_WORK = PATH_HOME + "embedding/"
	PATH_DATA = PATH_WORK + "data/"


# [0] Pre-processing: make my own dictionary.

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
with open(PATH_ILM_ACT + "genretags.txt", 'r') as tags_fp:
	for line in tags_fp:
		line_array = line.rstrip("\n").split(" ")
		tags_matrix.append(line_array)

np_tags_matrix = np.array(tags_matrix)
np.save(PATH_DATA + "tags_matrix.np", np_tags_matrix)

pdb.set_trace()





