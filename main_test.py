import platform
import os
import sys
import cPickle as cP

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


# [0] Pre-processing: make my own dictionary.

dict_id_path = {}
with open(PATH_ILM_META + r"collection10000-810(500)-5(1)-loglikgmm-wav.csv", 'r') as id_path_fp:
	line = id_path_fp.readline() # ignore the first line
	for line in id_path_fp: # ignore the first line
		line = line.rstrip("\n").strip('"')
		song_id, song_path = line.split('","') #	line:"444","e/audio/x-wav/eb/1508-04.01.wav"
		dict_id_path[int(song_id)] = song_path.split("/")[-1]

cP.dump(dict_id_path, )




