"""
This module remove spurious words
"""

from gensim.utils import simple_preprocess

def remove_spurious_words(input_collection: 'list[str]', deacc: 'bool', min_len: 'int') -> 'list[list[str]]':
  collection = input_collection.copy()
  
  # Process each word of each document
  # and remove accents and words with length less than min_len
  collection = [simple_preprocess(" ".join(d), deacc = deacc, min_len = min_len) for d in collection]
  
  return collection