"""
This module process the bigrams
"""
from gensim.models.phrases import Phrases, Phraser

def proccess_bigrams(input_collection: 'list[list[str]]', min_count: 'int', threshold: 'int') -> 'list[list]':
  collection = input_collection.copy()

  # Transform documento into bigrams and remove those who is less relevant
  phrases = Phrases(sentences=collection, min_count=min_count, threshold=threshold)
  bigram = Phraser(phrases)

  return [bigram[d] for d in collection]
