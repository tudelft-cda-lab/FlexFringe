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
r_alphabet = None # dict mapping str to [input space of network]
ALPHABET_NAME = None # must be set


def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  pass


def map_sequence(seq: list):
  """Maps the sequences from the flexfringe input (integer values starting from zero
  strings, depending on implementation) to the input as needed by the network.

  Args:
      seq (list): The sequence as a list.

  Returns:
      list() or equivalent sequence type: The sequence mapped so that the network can take it.
  """
  global r_alphabet
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

  seq = map_sequence(seq)
  pass


def get_alphabet(path_to_model: str):
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must a list of string objects. Flexfringe will take care of the internals.

  Side effects (must be implemented!): 
  1. If the alphabet is uninitialized (None), set the alphabet accordingly.
  2. We need to set the r_alphabet, because this one will map the strings as 
  requested by flexfringe back to the input space of the network.

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument (used to infer the path to the alphabet. Must be in same dir.).

  Returns:
      alphabet: list(str)
  """
  global r_alphabet
  global ALPHABET_NAME

  alphabet_dir = os.path.split(path_to_model)
  path_to_alphabet = os.path.join(alphabet_dir, ALPHABET_NAME)

  if r_alphabet is None:
    r_alphabet = pk.load(open(path_to_alphabet, "rb"))

  assert(r_alphabet is not None)
  r_alphabet = {str(key): value for key, value in r_alphabet.items()}
  return list(r_alphabet.keys())


if __name__ == "__main__":
  raise Exception("This is not a standalone script.")