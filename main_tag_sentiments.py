
# https://finnaarupnielsen.wordpress.com/2011/06/20/simplest-sentiment-analysis-in-python-with-af/ 
# --> looks cool, but it (labMT) only has valence.. and they are integers!

from constants import *
from environments import *
import numpy as np
import os
import cPickle as cP
import pdb

class Mood_Sentiment():
	def __init__(self):
		self.moodnames = [] # generic python list
		self.vads_list = [] # 
		self.vads_dict = {} # python dictionary 
		self.dist_mtx = None # will ba a 2-d numpy array

	def add_item(self, key_to_add, val_to_add):
		'''Fill the lists and a dict with big_vad_dict which contains large vad sets of sentiments'''
		self.moodnames.append(key_to_add)
		self.vads_list.append(val_to_add)
		self.vads_dict[key_to_add] = val_to_add

	def compute_dist(self):
		self.dist_mtx = np.zeros((len(self.moodnames), len(self.moodnames)))
		for ind_i, vads_value_i in enumerate(self.vads_list):
			for ind_j, vads_value_j in enumerate(self.vads_list):
				pdb.set_trace()
				self.dist_mtx[ind_i, ind_j] = np.linalg.norm(np.array(vads_value_i)-np.array(vads_value_j)) 

	def get_nearest_by_moodname(self, moodname, num_word=10):
		if moodname not in self.moodnames:
			print '--- i dont have that moodname ---'
			return None
		return self.get_nearest_by_ind(self.moodnames.index(moodname), num_word)

	def get_nearest_by_ind(self, ind, num_word=10):
		if ind >= len(self.moodnames):
			print '--- too large index ---'
			return None
		
		sorted_index = np.argsort(self.dist_mtx[ind, :])
		num_word = min(num_word, len(self.moodnames))
		words_to_return = []
		for i in xrange(num_word):
			words_to_return.append(self.moodnames[sorted_index[i]])
		return words_to_return


if __name__=='__main__':

	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	#,Word,V.Mean.Sum,V.SD.Sum,V.Rat.Sum,A.Mean.Sum,A.SD.Sum,A.Rat.Sum,D.Mean.Sum,D.SD.Sum,D.Rat.Sum,V.Mean.M,V.SD.M,V.Rat.M,V.Mean.F,V.SD.F,V.Rat.F,A.Mean.M,A.SD.M,A.Rat.M,A.Mean.F,A.SD.F,A.Rat.F,D.Mean.M,D.SD.M,D.Rat.M,D.Mean.F,D.SD.F,D.Rat.F,V.Mean.Y,V.SD.Y,V.Rat.Y,V.Mean.O,V.SD.O,V.Rat.O,A.Mean.Y,A.SD.Y,A.Rat.Y,A.Mean.O,A.SD.O,A.Rat.O,D.Mean.Y,D.SD.Y,D.Rat.Y,D.Mean.O,D.SD.O,D.Rat.O,V.Mean.L,V.SD.L,V.Rat.L,V.Mean.H,V.SD.H,V.Rat.H,A.Mean.L,A.SD.L,A.Rat.L,A.Mean.H,A.SD.H,A.Rat.H,D.Mean.L,D.SD.L,D.Rat.L,D.Mean.H,D.SD.H,D.Rat.H
	#1,aardvark,6.26,2.21,19,2.41,1.4,22,4.27,1.75,15,6.18,1.66,11,6,2.94,7,3,1.41,8,2.07,1.33,14,4,1.58,5,4.4,1.9,10,6.12,2.03,8,6.36,2.42,11,2.56,1.74,9$
	
	# Build a dictionary using csv file.
	print '--- load a big dictionary ---'
	if os.path.exists(PATH_DATA + FILE_DICT["sentiment_big_dict"]):
		vad_dict = cP.load(open(PATH_DATA + FILE_DICT["sentiment_big_dict"], 'r'))
	else:
		ugent_csv_file = "Ratings_Warriner_et_al.csv"
		vad_dict = {}
		with open(PATH_SENTI + ugent_csv_file, 'r') as f_read:
			temp = f_read.readline() # ignore the first line
			for line in f_read:
				line_split = line.split(',') # 0:index, 1:word, 2:valence mean, 5:arousal mean, 8:dominance mean
				vad_dict[line_split[1]] = np.array([line_split[2], line_split[5], line_split[8]])

		cP.dump(vad_dict, open(PATH_DATA + FILE_DICT["sentiment_big_dict"], 'w'))

	print '--- small dictionary+etc for mood tags only'
	mood_sentiment = Mood_Sentiment()
	for moodname in moodnames:
		if moodname in vad_dict:
			mood_sentiment.add_item(moodname, vad_dict[moodname])
	mood_sentiment.compute_dist() # distances for all pairs
	cP.dump(mood_sentiment, open(PATH_DATA + FILE_DICT["mood_sentiment"], 'w'))

	for moodname in mood_sentiment.moodnames:
		print mood_sentiment.get_nearest_by_moodname(moodname, num_word=10)

	pdb.set_trace()




	
	



