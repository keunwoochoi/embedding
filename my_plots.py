import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def export_history(loss, val_loss, acc=None, val_acc=None,  out_filename):
	'''
	subplots; acc-valacc on top, loss-val_loss on bottom
	'''
	if acc:
		pass
	else:

		f = plt.plot(loss)
		plt.plot(val_loss)
		
		ax2.set_title('Loss')
		# legend = plt.legend(loc='lower left', shadow=False)

	plt.savefig(out_filename)
	plt.close()
