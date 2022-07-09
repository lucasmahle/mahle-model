"""
Model MahleModel

TFIDFK Model process the collection to get TF-IDF corpus
and exposes features methods to retrive topics or documents
"""

from mahlemodel.collection import clear_stop_words_and_get_lemma, remove_spurious_words, proccess_bigrams, generate_dictionary, generate_bow, generate_tfidf_model
from mahlemodel.cluster import get_kmeans_data, generate_clusters, calculate_tfidf_avg, get_relevants_words

from gensim.interfaces import TransformedCorpus

class MahleModel:
  def __init__(self, collection: 'list[str]', num_topics=0, num_words_per_topic=0):
    self.__collection = collection.copy()
    self.__num_topics = num_topics
    self.__num_words_per_topic = num_words_per_topic
    self.__corpus_tfidf = None
    
    if self.__num_topics > 0:
      self.__run_model()

  def __run_model(self):
    self.__proccess_collection()
    self.__proccess_cluster()
  
  def __proccess_collection(self):
    collection = clear_stop_words_and_get_lemma(self.__collection)
    collection = remove_spurious_words(collection, deacc=True, min_len=3)
    self.bigrams = collection = proccess_bigrams(collection, min_count=10, threshold=5)
    self.dictionary = generate_dictionary(collection, no_below=.1, no_above=.8, keep_n=10000)
    corpus_bow = generate_bow(collection, self.dictionary)
    tfidf = generate_tfidf_model(corpus_bow, self.dictionary)

    self.__corpus_tfidf: 'TransformedCorpus' = tfidf[corpus_bow]

  def __proccess_cluster(self):
    self.__kmeans_data = get_kmeans_data(self.dictionary, self.__corpus_tfidf)
    print(len(self.__kmeans_data))
    print(len(self.__kmeans_data[0]))
    self.__kmeans = generate_clusters(self.__num_topics, self.__kmeans_data)
    print(len(self.__kmeans.cluster_centers_))
    print(len(self.__kmeans.cluster_centers_[0]))

    tfidf_cluster = calculate_tfidf_avg(self.__corpus_tfidf, self.__kmeans, self.__num_topics, self.dictionary)
    
    self.__dic_cluster_topics = get_relevants_words(num_words=self.__num_words_per_topic, tfidf_cluster=tfidf_cluster, dictionary=self.dictionary)

  def set_num_relevants_words(self, num_words_per_topic: 'int'):
    self.__num_words_per_topic = num_words_per_topic

  def set_num_topics(self, num_topics: 'int'):
    self.__num_topics = num_topics
    self.__proccess_cluster()

  def get_topics(self):
    return [self.__dic_cluster_topics[x] for x in self.__dic_cluster_topics]

  def get_topics_by_document(self, document: 'int|str'):
    if isinstance(document, int):
      document_index = document
      # checkar se existe index no array
      document = self.__collection[document_index]
      ddd = self.__kmeans_data[document_index]

      print(self.__kmeans.predict(ddd))

      # Pegar o node dentro do kmeans
      # Buscar todos centroids
      # Calcular distância euclidiana
      # Gerar lista 
      # Filtrar os mais relevantes

    """
    pegar o doc no kmeans
    pegar a distância desse doc para todos os centroids (euclidiana)
    passar param de threshold pra retornar só os mais relevantes
    """
    return []

  def get_document_topics_by(self, topic: 'int'):
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