"""
Test for the connector on our simple benchmarks.
"""

import os

import numpy as np
import pickle as pk

import tensorflow.keras
from tensorflow.keras.models import load_model

ALPHABET_NAME = "alphabet_mapping.pk" # this is the name of the alphabet. Assumed to be in the same directory as the model

model = None
alphabet = None
r_alphabet = None

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
  global alphabet

  if(len(seq) == 0):
    return 0.1
  seq = np.array(seq)
  seq_one_hot = np.zeros((1, seq.shape[0], len(alphabet)))
  for i, sym in enumerate(seq):
    seq_one_hot[0, i, sym] = 1
  y_pred = model.predict(seq_one_hot, verbose=0)
  return float(y_pred[0])


def get_alphabet(path_to_model: str):
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must a list of int objects, representing the possible inputs of the network. 
  Flexfringe will take care of the internals.

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument (used to infer the path to the alphabet. Must be in same dir.).

  Returns:
      alphabet: list(int): The possible inputs the network can take.
  """
  global alphabet
  global ALPHABET_NAME

  alphabet_dir, _ = os.path.split(path_to_model)
  path_to_alphabet = os.path.join(alphabet_dir, ALPHABET_NAME)

  if alphabet is None:
    alphabet = pk.load(open(path_to_alphabet, "rb"))

  assert(alphabet is not None)
  return list(alphabet.values()) # a bit of legacy here, we can change that in the training scripts


if __name__ == "__main__":
    raise Exception("This script is not meant as a standalone.")