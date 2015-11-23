""" Apply word2vec to all tags (100 from ilm10k) and 
"""
from constants import *
from environments import *
import gensim
import logging
import os
import sys
import multiprocessing
import cPickle as cP

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
# http://williambert.online/2012/05/relatively-quick-and-easy-gensim-example-code/
# background_corpus = gensim.TextCorpus()

def generate_lines(corpus):
    for index, text in enumerate(corpus.get_texts()):
        if index < max_sentence or max_sentence==-1:
            yield text
        else:
            break

def init_logger():
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)

	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	return logger

def prepare_wiki_text():
	# [1] prepare wiki text
	logger = init_logger()
	inp = 'enwiki-20151102-pages-articles.xml.bz2'
	outp = 'wiki.en.text'
	if os.path.exists(PATH_WIKI + outp):
		if os.path.getsize(PATH_WIKI+outp) == 0:
			print 'There is already wiki.en.text in the path, but the size is 0, which is strange.'
			print 'Therfore I will just proceed'
			os.remove(PATH_WIKI + outp)
		else:
			print 'prepare_wiki_text : already done. remove %s if you want to run it again.' % outp
			return

	output = open(PATH_WIKI + outp, 'w')
	wiki = WikiCorpus(PATH_WIKI + inp, lemmatize=False)
	space = " "
	i = 0
	for text in wiki.get_texts():
		output.write(space.join(text) + "\n")
		i += 1
		if (i % 10000 == 0):
			logger.info("Saved " + str(i) + " articles.")

	output.close()
	logger.info("Finished saved " + str(i) + " articles.")	

def train_word2vec_model():

	logger = init_logger()
	dim = 200
	inp = 'wiki.en.text'
	outp = 'wiki.en.text.model_%d_dim' % dim
	outp2= 'wiki.en.text.vector_%d_dim' % dim
	if os.path.exists(PATH_WIKI + outp2):
		print 'train_word2vec_model : already done. remove %s if you want to train it again.' % outp2
		model = gensim.models.Word2Vec.load_word2vec_format(PATH_WIKI + outp2, binary=False) # takes few minutes.
		return model

	model = Word2Vec(LineSentence(PATH_WIKI + inp), size=dim, window=5, min_count=10, workers=multiprocessing.cpu_count())

	# model.init_sims(replace=True) 
	model.save(PATH_WIKI + outp)
	model.save_word2vec_format(PATH_WIKI + outp2, binary=False)
	print "Fin.; training word2vec model."
	return model

def get_embeddings(model):
	"""input: word2vec model from gensim."""
	if os.path.exists(PATH_DATA + FILE_DICT["mood_embeddings"]):
		return = cP.load(open(PATH_DATA + FILE_DICT["mood_embeddings"]))

	moodnames = cP.load(open(PATH_DATA + FILE_DICT["moodnames"], 'r')) #list, 100
	embeddings = {}
	for moodname in moodnames:
		try:
			embeddings[moodname]=model[moodname]
		except:
			pass
	cP.dump(embeddings, open(PATH_DATA + FILE_DICT["mood_embeddings"], 'w'))
	return embeddings

def reduce_dims(embeddings):
	"""input: embeddings dictionary for mood.
				key: moodname
				value: 
	return: dictionary in same size as input with reduced dims of vectors.
	"""
	width = embeddings[embeddings.keys()[0]].shape[0]
	height = len(embeddings) # number of moods w/ vector representation
	# load data
	big_mtx = np.zeros((height, width))
	row_ind = 0
	keys = []
	for key in embeddings:
		keys.append(key)
		big_mtx[row_ind,:] = embeddings[key]
		row_ind += 1
	# dim reduction
	from sklearn.decomposition import PCA
	pca = PCA(n_components=20)
	# return
	return pca.fit_transform(big_mtx)

def cluster_embeddings(data_mtx, num_clusters):
	"""

	"""
	from sklearn.cluster import KMeans
	kmeans = KMeans(num_clusters)
	return kmeans.fit_predict(data_mtx)


if __name__ == "__main__":

	'''
	bow_corpus = gensim.corpora.MmCorpus(PATH_WIKI + "_tfidf.mm", background_corpus)
	# http://stackoverflow.com/questions/23735576/gensim-train-word2vec-on-wikipedia-preprocessing-and-parameters
	corpus = gensim.corpora.WikiCorpus(PATH_WIKI + 'enwiki-20151102-pages-articles.xml.bz2', dictionary=False)
	max_sentence = -1

	# dictionary = gensim.corpora.Dictionary.load_from_text(PATH_WIKI + '_wordids.txt')
	model = gensim.models.word2vec.Word2Vec()
	model.build_vocab(generate_lines())
	model.train(generate_lines(), chunksize=500)

	### https://radimrehurek.com/gensim/wiki.html
	# load id->word mapping (the dictionary)
	id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
	# load corpus iterator
	mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
	print(mm) # MmCorpus(3931787 documents, 100000 features, 756379027 non-zero entries)
	'''

	# http://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
	# [1]
	prepare_wiki_text()

	# [2] training a word2vec model. Or load it if it's done before.
	model = train_word2vec_model()

	# [3] use the model to get each tag's embedding vector
	embeddings = get_embeddings(model) # 96-by-200. 96:#tags, 200: dimension of vector
	
	# [4] dim reduction of embeddings using PCA
	reduced_embeddings = reduce_dims(embeddings) # 96-by-20

	# [5] clustering with K-means
	clusters_reduced_embeddings = cluster_embeddings(data_mtx=reduced_embeddings, num_clusters=10):
	count_dict = dict((i, list(clusters_reduced_embeddings).count(i)) for i in clusters_reduced_embeddings)
	

