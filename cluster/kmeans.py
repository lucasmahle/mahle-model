from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.interfaces import TransformedCorpus
from sklearn.cluster import KMeans

def get_kmeans_data(dictionary: 'Dictionary', corpus_tfidf: 'TransformedCorpus'):
  num_docs = dictionary.num_docs
  num_words = len(dictionary.keys())

  # Extracts the dense matrix (few elements are nulls)
  corpus_tfidf_dense = corpus2dense(corpus=corpus_tfidf, num_terms=num_words, num_docs=num_docs)

  # Get transposed matrix
  return corpus_tfidf_dense.T

def generate_clusters(n_clusters, kmeans_data):
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(kmeans_data)
  return kmeans