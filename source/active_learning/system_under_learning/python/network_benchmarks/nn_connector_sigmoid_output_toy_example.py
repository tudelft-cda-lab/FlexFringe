"""
Test for the connector on our simple benchmarks.
"""

import numpy as np
import pickle as pk

import tensorflow.keras
from tensorflow.keras.models import load_model

MODEL_NAME = "nn_scripts/model_problem_1.h5"
ALPHABET_NAME = "nn_scripts/alphabet_mapping.pk"

model = None
alphabet = None


def load_model():
  """Load a model and writes it into the global model-variable
  """
  global model
  global MODEL_NAME
  model = load_model(MODEL_NAME)
  print("Model loaded successfully")


def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
  assert(model is not None)
  
  seq_mapped = [alphabet[symbol] for symbol in seq] # TODO: resolve the mismatch that can happen in between the two alphabets here, set alph in flexfringe
  seq_mapped = np.array(seq_mapped)

  seq_one_hot = np.zeros((1, seq_mapped.shape[0], len(alphabet)))
  for i, sym in enumerate(seq):
    seq_one_hot[0, i, alphabet[sym]] = 1
  
  y_pred = model.predict(seq_one_hot)
  return float(y_pred[0])


def get_alphabet():
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must be of dict()-type, mapping input alphabet of type str() to internal 
  representation in integer values.

  Side effect (must be implemented!): If the alphabet is uninitialized (None), set the alphabet accordingly.

  alphabet: dict() [str : int]
  """
  global alphabet
  global ALPHABET_NAME

  if alphabet is None:
    alphabet = pk.load(open(ALPHABET_NAME, "rb"))

  assert(alphabet is not None)
  return alphabet


if __name__ == "__main__":
    raise Exception("This script is not meant as a standalone.")