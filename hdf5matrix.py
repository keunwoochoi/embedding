''' HDF5Martrix - a interface for hdf5 file.
copied/pasted from keras + small modifications
'''


from collections import defaultdict
import h5py
import numpy as np


class HDF5Matrix():
	def __init__(self, datapath, dataset, start, end, normalizer=None):
		self.refs = defaultdict(int) # MODI
		if datapath not in list(self.refs.keys()):
			# print 'Init with hdf path: %s'%datapath
			f = h5py.File(datapath)
			self.refs[datapath] = f
		else:
			f = self.refs[datapath]
		self.start = start
		self.end = end
		self.data = f[dataset]
		self.normalizer = normalizer

	def __len__(self):
		return self.end - self.start

	def __getitem__(self, key):
		if isinstance(key, slice):
			if key.stop + self.start <= self.end:
				idx = slice(key.start+self.start, key.stop + self.start)
			else:
				raise IndexError
		elif isinstance(key, int):
			if key + self.start < self.end:
				idx = key+self.start
			else:
				raise IndexError
		elif isinstance(key, np.ndarray):
			if np.max(key) + self.start < self.end:
				idx = (self.start + key).tolist()
			else:
				raise IndexError
		elif isinstance(key, list):
			if max(key) + self.start < self.end:
				idx = [x + self.start for x in key]
			else:
				raise IndexError
		
		if self.normalizer is not None:
			return self.normalizer(self.data[idx])
		else:
			return self.data[idx]

	@property
	def shape(self):
		ret = []
		ret.append(self.end - self.start)
		ret = ret + [dim for dim in list(self.data.shape[1:])]
		return tuple(ret) # MODI
