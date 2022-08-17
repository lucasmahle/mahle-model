"""
Model MahleModel

TFIDFK Model process the collection to get TF-IDF corpus
and exposes features methods to retrive topics or documents
"""

from mahlemodel.collection import generate_dictionary, generate_bow, generate_tfidf_model, mahle_collection_transform
from mahlemodel.cluster import get_kmeans_data, generate_clusters, calculate_tfidf_avg, get_relevants_words

from gensim.interfaces import TransformedCorpus
import numpy as np

class MahleModel:
  def __init__(self, words_per_topic=0, dictionary=None):
    self.__num_topics = 0
    self.__num_words_per_topic = words_per_topic
    self.__corpus_tfidf = None
    self.__bow = []
    self.__tokenized_docs = []
    self.dictionary = dictionary
  
  def __proccess_cluster(self):
    self.__kmeans_data = get_kmeans_data(self.dictionary, self.__corpus_tfidf)
    self.__kmeans = generate_clusters(self.__num_topics, self.__kmeans_data)

    tfidf_cluster = calculate_tfidf_avg(self.__corpus_tfidf, self.__kmeans, self.__num_topics, self.dictionary)
    
    self.__dic_cluster_topics = get_relevants_words(num_words=self.__num_words_per_topic, tfidf_cluster=tfidf_cluster, dictionary=self.dictionary)

  def fit_collection(self, collection: 'list[str]'):
    """ 
    Transform collection before fit them
    """
    collection = mahle_collection_transform(collection.copy())
    self.fit(collection)

  def fit(self, collection: 'list[str]'):
    """ 
    Generate dictionary and calculate features over them
    """
    self.__tokenized_docs = self.__tokenized_docs + collection
    self.dictionary = generate_dictionary(self.dictionary, collection, no_below=.1, no_above=.8, keep_n=10000)
    corpus_bow = generate_bow(collection, self.dictionary)
    self.__bow = self.__bow + corpus_bow
    tfidf = generate_tfidf_model(self.__bow, self.dictionary)

    self.__corpus_tfidf: 'TransformedCorpus' = tfidf[self.__bow]

  def get_bow(self):
    return self.__bow

  def get_documents(self):
    return self.__tokenized_docs

  def set_num_relevants_words(self, num_words_per_topic: 'int'):
    self.__num_words_per_topic = num_words_per_topic

  def calculate_topics(self, num_topics: 'int'):
    self.__num_topics = num_topics
    self.__proccess_cluster()

  def get_topics(self):
    return [self.__dic_cluster_topics[x] for x in self.__dic_cluster_topics]

  def get_document_topics(self, document_index:'int', minimum_probability:'float'=None):
    """ Calculate topics of document

    Arguments:
        document_index: Index of document used to fit model
        minimum_probability: When this value is setted, the model will return topics
                             only when the topic probability is higher then
                             minimum_probability

    Returns:
        document_composition: List of sorted topics, from most relevant to less
                              relevant.
    """
    document_data = self.__kmeans_data[document_index]
    document_composition = []
    
    # Calculate distance for each cluster
    for i, cluster in enumerate(self.__kmeans.cluster_centers_):
      distance = np.linalg.norm(np.array(cluster) - np.array(document_data))
      document_composition.append((i, distance))

    # Normalize value
    topics_values = list(map(lambda x: x[1], document_composition))

    # inverse distance weighting
    idw_sum = sum(1 / np.array(topics_values))
    document_composition = [(dc[0], ((1/dc[1]) / idw_sum)) for dc in document_composition]

    # Filter minimum probability
    if minimum_probability is not None:
      document_composition = [dc for dc in document_composition if dc[1] >= minimum_probability]

    return sorted(document_composition, key=lambda x: x[1], reverse=True)

  def get_topic_document(self, topic: 'int'):
    """
    pega o centroid desse tópico
    busca todos documentos próximos desse documento
    passar param de threshold pra retornar só os mais relevantes
    """
    return []

"""
Outros métodos:
add documento no treinamento

obter topcs de doc que não foi utilizado no treinamento
"""