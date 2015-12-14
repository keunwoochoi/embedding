import platform
import os
import sys

device_name = platform.node()

if device_name.endswith('eecs.qmul.ac.uk') or device_name in ['octave', 'big-bird']:
	isServer = True
	isMac = False
	isDT = False

else:
	isServer = False
	if device_name in["KChoiMBPR2013.local", "lt91-51"]:
		isMac = True
		isDT = False
	elif device_name == "keunwoo-dt-ubuntu":
		isMac = False
		isDT = True

if isServer:
	print "THIS IS A SERVER NAMED %s" % device_name

	PATH_IMPORT = '/import/'	
	PATH_HOME = "/homes/kc306/"

elif isDT:
	print "You are using Ubuntu Desktop"
	PATH_IMPORT = '/mnt/c4dm/'
	PATH_HOME   = '/mnt/kc306home/'

elif isMac:
	print "Do not use Mac for computation!"

PATH_STFT = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_transformed/' + 'STFT/'
PATH_CQT  = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_transformed/' + 'CQT/'
PATH_WIKI = PATH_IMPORT + "c4dm-datasets/Wikipedia_dump/"
PATH_ILM = PATH_IMPORT + 'c4dm-01/ilm10k-dataset/'
PATH_ILM_ACT = PATH_ILM + 'act-coordinates/'
#PATH_ILM_AUDIO = PATH_ILM + 'ilmaudio/'
PATH_ILM_AUDIO = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k-audio-copied/'
PATH_ILM_META = PATH_ILM + 'metadata/'

PATH_WORK = PATH_HOME + "embedding/"
PATH_DATA = PATH_WORK + "data/"
PATH_MODEL= PATH_WORK + 'keras_models/'
PATH_SENTI= PATH_WORK + "sentiment/" 

PATH_IMAGES=PATH_WORK + 'images/'

PATH_FIGURE = PATH_WORK + 'figures/'
PATH_RESULTS= PATH_WORK + 'results/'

for path in [PATH_DATA, PATH_MODEL, PATH_SENTI, PATH_IMAGES, PATH_FIGURE, PATH_RESULTS]:
	if not os.path.exists(path):
		os.mkdir(path)

sys.path.append(PATH_HOME + 'modules/' + 'librosa/')
