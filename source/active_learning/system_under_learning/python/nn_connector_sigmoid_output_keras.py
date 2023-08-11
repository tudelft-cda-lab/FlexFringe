"""
Connects to a sigmoid output network, i.e. a network with 
binary output.
"""

import os

import numpy as np
import pickle as pk

import tensorflow.keras
from tensorflow.keras.models import load_model

ALPHABET_NAME = "alphabet_mapping.pk" # this is the name of the alphabet. Assumed to be in the same directory as the model

model = None
alphabet = None

def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  model = load_model(path_to_model)
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
  
  y_pred = model.predict(seq_one_hot, verbose=0)
  return float(y_pred[0])


def get_alphabet(path_to_model: str):
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must be of dict()-type, mapping input alphabet of type str() to internal 
  representation in integer values.

  Side effect (must be implemented!): If the alphabet is uninitialized (None), set the alphabet accordingly.

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument (used to infer the path to the alphabet. Must be in same dir.).

  Returns:
      alphabet: dict() [str : int]
  """
  global alphabet
  global ALPHABET_NAME

  alphabet_dir, _ = os.path.split(path_to_model)
  path_to_alphabet = os.path.join(alphabet_dir, ALPHABET_NAME)

  if alphabet is None:
    alphabet = pk.load(open(path_to_alphabet, "rb"))

  assert(alphabet is not None)
  return alphabet

if __name__ == "__main__":
    raise Exception("This script is not meant as a standalone.")