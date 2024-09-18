"""
This module generate info in order to generate TF-IDF data
"""
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

def generate_dictionary(current_dictionary = None, collection: 'list[list]' = [], no_below: 'float' = 0, no_above: 'float' = 0, keep_n: 'int' = 0) -> 'Dictionary':
  dictionary = Dictionary(collection)

  # Merge current dictionary
  if current_dictionary is not None:
    dictionary.merge_with(current_dictionary)

  # Filter dictionary considering % of the word recorrency and most frequency words
  dictionary.filter_extremes(no_below=len(collection) * no_below, no_above=no_above, keep_n=keep_n)

  return dictionary

def generate_bow(collection: 'list[list]', dictionary: 'Dictionary') -> 'list':
  return [dictionary.doc2bow(d) for d in collection]

def generate_tfidf_model(corpus_bow: 'list', dictionary: 'Dictionary') -> 'TfidfModel':
  return TfidfModel(corpus_bow, dictionary=dictionary)