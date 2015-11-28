import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def export_history(loss, val_loss, acc=None, val_acc=None, out_filename='history.png'):
	'''
	subplots; acc-valacc on top, loss-val_loss on bottom
	'''
	if acc:
		pass
	else:

		f = plt.plot(loss)
		plt.plot(val_loss)
		
		# legend = plt.legend(loc='lower left', shadow=False)

	plt.savefig(out_filename)
	plt.close()


def save_weight_as_image(model, save_path = '', filename_prefix = '', normalize='local', mono=True):
	'''input: keras model variable and strings.
	save_path: path to save output image
	filename_prefix: prefix of the image file name.
	* It save png image of each layer's weights (image patches) visualisation
	 '''
	import numpy as np

	def make_mosaic(imgs, border=1):
	    """
	    Given a set of images with all the same shape, makes a
	    mosaic with nrows and ncols
	    modification: compute nrows/ncols implicitely
	    imgs: 3D tensor shaped as [num_image, height, width]
	    normalize: 'local', 'global', 'none'
	    """
	    import numpy.ma as ma

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
			
	from scipy.misc import imsave
	layerind = -1
	#print model.layers[0].W.get_value(borrow=True)[0,0,:,:]
	for layer in model.layers:
		layerind = layerind + 1
		g = layer.get_config()
		if not g['name'] == 'Convolution2D':
			continue # conv 2d layer only
		# W = layer.get_weights()[0] # tensor. same as layer.W.get_value(borrow=True)
		W = layer.W.get_value(borrow=True)
		#W = np.squeeze(W)
		'''
		for ind in xrange(W.shape[1]):
			W = W[:,ind,:,:]
		'''
		W.shape
		pdb.set_trace()
		if mono:
			ind = 0
			W = W[:,ind,:,:]
			filename = 'weights_' + repr(layerind) + '_' + filename_prefix + '_' + repr(ind) + '.png'
			mosaic = make_mosaic(imgs=W, border=2)
			imsave(save_path + filename, mosaic)

		else:
			ind = 0
			W = W[:,ind,:,:]
			filename = 'weights_left_' + repr(layerind) + '_' + filename_prefix + '_' + repr(ind) + '.png'
			mosaic = make_mosaic(imgs=W, border=2)
			imsave(save_path + filename, mosaic)

			ind = 1
			W = W[:,ind,:,:]
			filename = 'weights_righ_' + repr(layerind) + '_' + filename_prefix + '_' + repr(ind) + '.png'
			mosaic = make_mosaic(imgs=W, border=2)
			imsave(save_path + filename, mosaic)




