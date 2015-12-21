""" To predict tags! using ilm10k data, stft or cqt representation, 
"""
#import matplotlib
#matplotlib.use('Agg')
from constants import *
from environments import *
import numpy as np

import keras
import os
import pdb
import my_keras_models
import my_keras_utils
import my_utils
import cPickle as cP
import time
import sys
import my_plots

def print_usage_and_die():
	print 'python filename num_of_epoch(int) num_of_train_song(int) tf_type model_type num_of_layers'
	print 'ex) $ python main_learn_tag.py 200 5000 cqt vgg classification 4 5 6'
	sys.exit()

if __name__ == "__main__":
	
	if len(sys.argv) < 7:
		print_usage_and_die()

	nb_epoch = int(sys.argv[1])
	num_train_songs = int(sys.argv[2])
	tf_type = sys.argv[3]
	model_type = sys.argv[4]
	if sys.argv[5].lower() in ['reg', 'regression']:
		isRegression = True
		isClassification = False
	elif sys.argv[5].lower() in ['cla', 'classification']:
		isRegression = False
		isClassification = True

	num_layers_list = [int(sys.argv[i]) for i in xrange(6, len(sys.argv))]
	print '--- num_layers are ---'
	print num_layers_list
	# nb_epoch = 1
	clips_per_song = 3
	# label matrix
	dim_latent_feature = 3
	# label_matrix_filename = (FILE_DICT["mood_latent_matrix"] % dim_latent_feature)
	label_matrix_filename = (FILE_DICT["mood_latent_tfidf_matrix"] % dim_latent_feature) # tfidf is better!
	
	if os.path.exists(PATH_DATA + label_matrix_filename):
		label_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
	else:
		"print let's cook the mood-latent feature matrix"
		import main_prepare
		mood_tags_matrix = np.load(PATH_DATA + label_matrix_filename) #np matrix, 9320-by-100
		label_matrix = main_prepare.get_LDA(X=mood_tags_matrix, num_components=k, show_topics=False)
		np.save(PATH_DATA + label_matrix_filename, W)
	print 'size of mood tag matrix:'
	print label_matrix.shape

	# load dataset
	print "I'll take %d clips for each song." % clips_per_song
	train_x, train_y, valid_x, valid_y, test_x, test_y = my_utils.load_all_sets(label_matrix=label_matrix, 
																		clips_per_song=clips_per_song, 
																		num_train_songs=num_train_songs, 
																		tf_type=tf_type)
	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	# learning_id =  str(np.random.randint(999999))
	if isClassification:
		train_y = my_keras_utils.continuous_to_categorical(train_y)
		valid_y = my_keras_utils.continuous_to_categorical(valid_y)
		test_y  = my_keras_utils.continuous_to_categorical(test_y)

	for num_layers in num_layers_list:
		model_name = model_type + '_dim'+str(dim_latent_feature)+'_'+sys.argv[1] +'epochs_' + sys.argv[2] + 'songs' + sys.argv[3] + '_' + str(num_layers) + 'layers'
		model_name_dir = model_name + '/'
		fileout = model_name + '_results'
		print "="*60
		print model_name
		print "="*60

		if not os.path.exists(PATH_MODEL + model_name_dir):
			os.mkdir(PATH_MODEL + model_name_dir)
		if not os.path.exists(PATH_IMAGES + model_name_dir):
			os.mkdir(PATH_IMAGES + model_name_dir)
		start = time.time()
		print "--- going to build a keras model with height:%d, width:%d, num_labels:%d" % (train_x.shape[2], train_x.shape[3], train_y.shape[1])
	 	if isRegression:
	 		print '--- ps. this is a regression task. ---'
	 		model = my_keras_models.build_regression_convnet_model(height=train_x.shape[2], width=train_x.shape[3], 
	 																num_labels=train_y.shape[1], num_layers=num_layers, 
	 																model_type=model_type)
		else:
			print '--- ps. this is a classification task. ---'
			model = my_keras_models.build_classification_convnet_model(height=train_x.shape[2], width=train_x.shape[3], 
																		num_labels=train_y.shape[1], num_layers=num_layers, 
																		model_type=model_type)		
	 	until = time.time()
	 	print "--- keras model was built, took %d seconds ---" % (until-start)

		#prepare callbacks
		checkpointer = keras.callbacks.ModelCheckpoint(filepath=PATH_MODEL + model_name_dir +"weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
														verbose=1, save_best_only=False)
		weight_image_saver = my_keras_utils.Weight_Image_Saver(model_name_dir)
		history = my_keras_utils.History_Regression_Val()
		if isRegression:
			early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)
		else:
			early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)
		#train!
		my_plots.save_model_as_image(model, save_path=PATH_IMAGES+model_name_dir, filename_prefix='INIT_', 
									normalize='local', mono=False)
		predicted = model.predict(train_x, batch_size=16)
		np.save(PATH_RESULTS + fileout + '_predicted_and_truths_init.npy', [predicted, train_y])
		if isRegression:
			model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=32, nb_epoch=nb_epoch, 
						show_accuracy=False, verbose=1, 
						callbacks=[history, early_stopping, weight_image_saver, checkpointer])
		else:
			model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=16, nb_epoch=nb_epoch, 
						show_accuracy=True, verbose=1, 
						callbacks=[history, early_stopping, weight_image_saver, checkpointer])
		# model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=40, nb_epoch=nb_epoch, show_accuracy=False, verbose=1, callbacks=[history, early_stopping, weight_image_saver])
		#test
		loss_testset = model.evaluate(test_x, test_y, show_accuracy=False)
		predicted = model.predict(test_x, batch_size=40)
		#save results
		model.save_weights(PATH_MODEL + model_name_dir + ('final_after_%d.keras' % nb_epoch), overwrite=True) 
		
		np.save(PATH_RESULTS + fileout + '_history.npy', history.val_losses)
		np.save(PATH_RESULTS + fileout + '_loss_testset.npy', loss_testset)
		np.save(PATH_RESULTS + fileout + '_predicted_and_truths_final.npy', [predicted, test_y])
		
		my_plots.export_history(history.losses, history.val_losses, acc=None, val_acc=None, out_filename=PATH_RESULTS + fileout + '.png')
		my_plots.save_model_as_image(model, save_path=PATH_IMAGES+model_name_dir, filename_prefix='', 
									normalize='local', mono=False)
		
	# figure_filepath = PATH_FIGURE + model_name + '_history.png'
