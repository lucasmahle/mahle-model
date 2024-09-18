import numpy as np

def calculate_tfidf_avg(corpus_tfidf, kmeans, n_clusters, dictionary):
  # docs_by_cluster = {0: [4, 7, 8], 1: [1, 3, 6], 2: [2, 5]}
  docs_by_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(n_clusters)}
  tfidf_cluster_matrix = {}

  for k in docs_by_cluster:
    # index_doc_cluster = [4, 7, 8]
    index_doc_cluster = docs_by_cluster[k]
    tfidf_cluster_avg = {} # objeto com index_palavra: media; esse objeto é instanciado para cada cluster
    
    for index_doc in index_doc_cluster:
      # Matriz tfidf do doc index_doc. ex: 4
      # corpus_tfidf[index] => lista com (id_palavra, tfidf)
      tfidf_doc = corpus_tfidf[index_doc]
          
      # Percorrer palavras do bow
      for word_i in dictionary:
        # Inicia a média da palavra do cluster com 0
        if not word_i in tfidf_cluster_avg: tfidf_cluster_avg[word_i] = None
        
        # Se a palavra i não consta no tfidf do documento, ignora
        if not np.isin(word_i, [word_tfidf[0] for word_tfidf in tfidf_doc]): continue
        
        # word_tfidf = (id_word, tfidf_word)
        # word_tfidf = (0, 1)
        # Busca no tfidf do documento o valor tfidf referente a palavra
        tfidf_value = [word_tfidf[1] for word_tfidf in tfidf_doc if word_tfidf[0] == word_i][0]
        
        # Calcula média da palavra
        tfidf_cluster_avg[word_i] = (tfidf_cluster_avg[word_i] + tfidf_value) / 2 if tfidf_cluster_avg[word_i] is not None else tfidf_value
    
    # Troca None por 0
    for key in tfidf_cluster_avg: tfidf_cluster_avg[key] = tfidf_cluster_avg[key] if tfidf_cluster_avg[key] is not None else 0

    # Seta a matriz de média tfidf por palavra ao cluster
    tfidf_cluster_matrix[k] = tfidf_cluster_avg

  return tfidf_cluster_matrix

def get_relevants_words(num_words, tfidf_cluster, dictionary):
  cluster_topics = {}

  for k in tfidf_cluster:
    sorted_cluster_tfidf = dict(sorted(tfidf_cluster[k].items(), key = lambda item: item[1], reverse = True))
    cluster_topics[k] = []
    
    i = 0
    for word_id in sorted_cluster_tfidf:
      word = dictionary[word_id]
      prob = sorted_cluster_tfidf[word_id]
      cluster_topics[k].append((word, prob))

      if i == num_words: break
      i += 1
    
  return cluster_topics
