import platform
import os
import sys

device_name = platform.node()

if device_name.startswith('ewert-server'):
	isMacPro = True
	isServer = False
	isMacbook= False
	isDT1     = False
	isDT2 	 = False
elif device_name in["KChoiMBPR2013.local", "KChoi.MBPR.2013.home", "lt91-51", 'lt91-51.eecs.qmul.ac.uk']:
	isMacPro = False
	isServer = False
	isMacbook = True
	isDT1 = False
	isDT2 = False

elif device_name.endswith('eecs.qmul.ac.uk') or device_name in ['octave', 'big-bird']:
	isMacPro = False
	isServer = True
	isMacbook= False
	isDT1 	 = False
	isDT2 	 = False

else:
	isMacPro = False
	isServer = False
	if device_name == "keunwoo-dt-ubuntu":
		isMacbook = False
		isDT1 = True
		isDT2 = False
	elif device_name == "keunwoo-dt2":
		isMacbook= False
		isDT1 = False
		isDT2 = True

if isMacPro:
	print "This is MacPro in CS.319"
	PATH_IMPORT = '/Users/keunwoo/mnt/c4dm/'
	PATH_HOME   = '/Users/keunwoo/mnt/kc306home/'

elif isServer:
	print "THIS IS A SERVER NAMED %s" % device_name

	PATH_IMPORT = '/import/'	
	PATH_HOME = "/homes/kc306/"

elif isDT1:
	print "You are using Ubuntu Desktop"
	PATH_IMPORT = '/mnt/c4dm/'
	PATH_HOME   = '/mnt/kc306home/'

elif isMacbook:
	print "Do not use MacbookPro for computation!...I hope."
	PATH_IMPORT = '/Users/gnu/mnt/c4dm/'
	PATH_HOME   = '/Users/gnu/GoogleDrive/phdCodes/'


PATH_STFT = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_transformed/' + 'STFT/'
PATH_CQT  = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_transformed/' + 'CQT/'

PATH_WIKI = PATH_IMPORT + "c4dm-datasets/Wikipedia_dump/"
PATH_ILM = PATH_IMPORT + 'c4dm-01/ilm10k-dataset/'

PATH_HDF = PATH_ILM + 'hdf/'

PATH_ILM_ACT = PATH_ILM + 'act-coordinates/'
#PATH_ILM_AUDIO = PATH_ILM + 'ilmaudio/'
PATH_ILM_AUDIO = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_copy/'

PATH_MFCC = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/mfcc20/'
PATH_CHROMA = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/chroma/'
PATH_HGRAM = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/harmonigram/'
PATH_PGRAM = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/pitchgram/'

PATH_CQT_H = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/cqt_harmony/'
PATH_CQT_P = PATH_IMPORT + 'c4dm-04/keunwoo/ilm10k_audio_features/cqt_percussive/'

PATH_ILM_META = PATH_ILM + 'metadata/'

if isMacbook:
	PATH_WORK = PATH_HOME + "embedding_tag/"
else:
	PATH_WORK = PATH_HOME + "embedding/"
PATH_DATA = PATH_WORK + "data/"
PATH_MODEL= PATH_WORK + 'keras_models/'
PATH_SENTI= PATH_WORK + "sentiment/" 

PATH_IMAGES=PATH_WORK + 'images/'

PATH_FIGURE = PATH_WORK + 'figures/'
PATH_RESULTS= PATH_WORK + 'results/'

for path in [PATH_DATA, PATH_MODEL, PATH_SENTI, PATH_IMAGES, 
             PATH_FIGURE, PATH_RESULTS]:
	if not os.path.exists(path):
		os.mkdir(path)

if isMacbook:
	pass
else:

	for path in [PATH_MFCC, PATH_CHROMA,
	             PATH_HGRAM, PATH_PGRAM, PATH_CQT, PATH_STFT,
	             PATH_CQT_H, PATH_CQT_P, PATH_HDF]:
		if not os.path.exists(path):
			os.mkdir(path)

	# sys.path.append(PATH_HOME + 'modules/' + 'librosa/')
