"""
This module generate info in order to generate TF-IDF data
"""
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

def generate_dictionary(input_collection: 'list[list]', no_below: 'float', no_above: 'float', keep_n: 'int') -> 'Dictionary':
  collection = input_collection.copy()

  dictionary = Dictionary(collection)

  # Filter dictionary considering % of the word recorrency and most frequency words
  dictionary.filter_extremes(no_below=len(collection) * no_below, no_above=no_above, keep_n=keep_n)

  return dictionary

def generate_bow(collection: 'list[list]', dictionary: 'Dictionary') -> 'list':
  return [dictionary.doc2bow(d) for d in collection]

def generate_tfidf_model(corpus_bow: 'list', dictionary: 'Dictionary') -> 'TfidfModel':
  return TfidfModel(corpus_bow, dictionary = dictionary)