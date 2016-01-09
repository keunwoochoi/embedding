import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os

import pdb
from scipy.misc import imsave
from numpngw import write_png
import numpy.ma as ma

def export_history(loss, val_loss, acc=None, val_acc=None, out_filename='history.png'):
	'''
	subplots of loss and acc
	'''
	if acc:
		f, (ax1, ax2) = plt.subplots(1,2)
		ax1.plot(loss)
		ax1.plot(val_loss)
		ax1.set_title('Loss')

		ax2.plot(acc)
		if val_acc:
			ax2.plot(val_acc)

		ax2.set_title('Accuracy')
	else:
		f = plt.plot(loss, linewidth=2)
		plt.plot(val_loss, '--')

	plt.savefig(out_filename)
	plt.close()

	#
	f = plt.plot(loss)
	plt.savefig('%s_loss_%2.4f.png' % (out_filename.split('.')[0], np.min(loss)))
	plt.close()
	
	f = plt.plot(val_loss)
	plt.savefig('%s_val_loss_%2.4f.png' % (out_filename.split('.')[0], np.min(val_loss)))
	plt.savefig(out_filename.split('.')[0] + '_val_loss.png')
	plt.close()

def make_mosaic(imgs, normalize, border=1):
	"""
	Given a set of images with all the same shape, makes a
	mosaic with nrows and ncols
	modification: compute nrows/ncols implicitely
	imgs: 3D tensor shaped as [num_image, height, width]
	normalize: 'local', 'global', 'none'
	"""
	

	nimgs = imgs.shape[0]
	imshape = imgs.shape[1:]
	#print nimgs
	#print imshape
	
	from numpy import ceil, sqrt

	if nimgs == 32:
		ncols = 8
		nrows = 4
	elif nimgs == 48:
		ncols = 8
		nrows = 6
	elif nimgs == 64:
		ncols = 8
		nrows = 8
	elif nimgs== 24:
		ncols = 6
		nrows = 4
	elif nimgs==96:
		ncols = 8
		nrows = 12
	elif nimgs==40:
		ncols = 8
		nrows = 5
	else:
		ncols = int(ceil(sqrt(nimgs)))
		nrows = int(ceil(float(nimgs) / ncols))
	'''
	mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
							ncols * imshape[1] + (ncols - 1) * border),
							dtype=np.float32)
	'''
	mosaic = np.zeros((nrows * imshape[0] + (nrows - 1) * border,
						ncols * imshape[1] + (ncols - 1) * border),
						dtype=np.float32)
	paddedh = imshape[0] + border
	paddedw = imshape[1] + border

	global_max = np.max(imgs)
	global_min = np.min(imgs)

	for i in xrange(nimgs):
		#print imgs[i]
		row = int(np.floor(i / ncols))
		col = i % ncols
		if normalize == 'local':
			img_min = np.min(imgs[i])
			img_max = np.max(imgs[i])
			imgs[i] = (imgs[i] - img_min) / (img_max - img_min)
		elif normalize == 'global':
			imgs[i] = (imgs[i] - global_min) / (global_max - global_min)
		#pdb.set_trace()
		mosaic[row * paddedh:row * paddedh + imshape[0],
			   col * paddedw:col * paddedw + imshape[1]] = imgs[i]
	#mosaic = 255 * mosaic # imsave want it to be 8-bit integer 
	return mosaic

def save_weights_changes_plot(weight_changes, save_path):
	num_layer = weight_changes.shape[1]
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.plot(weight_changes)
	lgd = ax.legend([('layer_%d'%idx) for idx in xrange(num_layer)], loc='best')
	ax.set_title('average weight changes amounts')
	fig.savefig(save_path + 'average_weight_changes.png')


def save_model_as_image(model, save_path = '', filename_prefix = '', normalize='local', mono=True):
	'''input: keras model variable and strings.
	save_path: path to save output image
	filename_prefix: prefix of the image file name.
	* It save png image of each layer's weights (image patches) visualisation
	 '''
			
	
	#print model.layers[0].W.get_value(borrow=True)[0,0,:,:]
	for layerind, layer in enumerate(model.layers):
		g = layer.get_config()
		if g['name'] == 'Convolution2D':
			# W = layer.get_weights()[0] # tensor. same as layer.W.get_value(borrow=True)
			W = layer.W.get_value(borrow=True)
			#W = np.squeeze(W)
			'''
			for ind in xrange(W.shape[1]):
				W = W[:,ind,:,:]
			'''
			save_weight_as_image(W, save_path, filename_prefix, normalize, mono, layerind)
		elif g['name'] == 'Dense':

			W = layer.W.get_value(borrow=True) # 
			save_histogram_as_image(W, save_path, filename_prefix, layerind)
	
def save_weight_as_image(W, save_path, filename_prefix, normalize, mono, layerind):
	'''W:weights
	save_path: path to save the images
	normlize: weather or not they would be normalised '''
	# if mono is True:
	ind = 0
	W = W[:,ind,:,:]
	mosaic = make_mosaic(imgs=W, normalize=normalize, border=2)
	#save
	folder_name = 'layer_%02.0d/' % layerind
	if not os.path.exists(save_path + folder_name):
		os.makedirs(save_path + folder_name)
	files_already_there = os.listdir(save_path + folder_name)
	files_already_there = [filename for filename in files_already_there if filename.endswith('.png')]
	
	# if len(files_already_there) == 0:
	# 	filename = 'weights_' + filename_prefix + '%02.0d_%03.0d.png' % (layerind, len(files_already_there))
	# 	imsave(save_path + folder_name + filename, mosaic)
	
	filename = 'weights_' + filename_prefix + '%02.0d_%03.0d.png' % (layerind, len(files_already_there))
	imsave(save_path + folder_name + filename, mosaic)

	# else:
		# ind = 0
		# W_left = W[:,ind,:,:]
		# filename = 'weights_left_' + repr(layerind) + '_' + filename_prefix + '_' + repr(ind) + '.png'
		# mosaic = make_mosaic(imgs=W_left, normalize=normalize, border=2)
#        mosaic = int(mosaic * 2**8) -- not working.
#        write_png(save_path + filename, mosaic)

		# ind = 1
		# W_right = W[:,ind,:,:]
		# filename = 'weights_righ_' + repr(layerind) + '_' + filename_prefix + '_' + repr(ind) + '.png'
		# mosaic = make_mosaic(imgs=W_right, normalize=normalize, border=2)
#        mosaic = int(mosaic * 2**8)
#		write_png(save_path + filename, mosaic)
def save_histogram_as_image(W, save_path, filename_prefix, layerind):
	''''''
	n, bins, patches = plt.hist(W.flatten(), 100)
	
	y = mlab.normpdf( bins, np.mean(W.flatten()), np.std(W.flatten()))
	l = plt.plot(bins, y, 'r--', linewidth=1)
	plt.xlabel('Coeffs')

	folder_name = 'layer_%02.0d/' % layerind
	if not os.path.exists(save_path + folder_name):
		os.makedirs(save_path + folder_name)
	files_already_there = os.listdir(save_path + folder_name)
	files_already_there = [filename for filename in files_already_there if filename.endswith('.png')]

	filename = 'weights_' + repr(layerind) + '_' + filename_prefix + '_' + repr(len(files_already_there)) + '.png'
	
	filename = filename_prefix + 'histogram_fc_layer_%02.0d_%03.0d.png' % (layerind, len(files_already_there))
	plt.savefig(save_path + folder_name + filename)
	plt.close()





