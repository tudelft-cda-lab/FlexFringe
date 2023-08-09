"""
This file is a template to connect the SUL to 
a neural network and is meant to capture and transfer
outputs and queries in between the two systems.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 31.7.23.
"""

import os
import pickle as pk

# we need these variables globally, as they represent the status of our program
model = None 

alphabet = None
ALPHABET_NAME = None # must be set

def load_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  pass

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
  assert(model is not None)
  pass

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

  alphabet_dir = os.path.split(path_to_model)
  path_to_alphabet = os.path.join(alphabet_dir, ALPHABET_NAME)

  if alphabet is None:
    alphabet = pk.load(open(path_to_alphabet, "rb"))

  assert(alphabet is not None)
  return alphabet

if __name__ == "__main__":
  raise Exception("This is not a standalone script.")