
from . import clear_stop_words_and_get_lemma, remove_spurious_words, process_bigrams

def mahle_collection_transform(collection: 'list[str]'):
  collection = clear_stop_words_and_get_lemma(collection)
  collection = remove_spurious_words(collection, deacc=True, min_len=3, max_len=15)
  collection = process_bigrams(collection, min_count=10, threshold=5)

  return collection