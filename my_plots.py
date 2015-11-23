import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def export_history(acc, val_acc, loss, val_loss, out_filename, net_name=None):
	'''
	subplots; acc-valacc on top, loss-val_loss on bottom
	'''
	x1 = range(len(acc))
	x2 = x1

	f, (ax1, ax2) = plt.subplots(2,1)
	ax1.plot(x1, acc, 'b', label='train_acc')
	ax1.plot(x1, val_acc, 'g', label='valid_acc')
	if net_name is None:
		ax1.set_title('Accuracy')
	else:
		ax1.set_title('Accuracy for ' + net_name)
	ax1.set_ylim(0,1)
	
	ax2.plot(x2, loss, 'b', label='train_loss')
	ax2.plot(x2, val_loss, 'g', label='valid_loss')
	ax2.set_title('Loss')
	# legend = plt.legend(loc='lower left', shadow=False)

	plt.savefig(out_filename)
	plt.close()
