"""
This module handle word's lemma, stopwords and puntions
"""
import spacy

def load_spacy():
  try:
    return spacy.load('en_core_web_md')
  except OSError:
    spacy.cli.download("en_core_web_md")
    return spacy.load('en_core_web_md')

def clear_stop_words_and_get_lemma(input_collection: 'list[str]') -> 'list[str]':
  collection = input_collection.copy()

  # Convert to lowercase
  collection = [d.lower() for d in collection]

  nlp = load_spacy()

  for i, d in enumerate(collection):
    # Get tokens from document
    d = nlp(d)
    
    # Remove stopwords, puntions and underscore lemma
    d = [w.lemma_ for w in d if not(w.is_stop == True or w.is_punct == True or '_' in w.lemma_)]

    # Set converted collection
    collection[i] = d
  
  return collection