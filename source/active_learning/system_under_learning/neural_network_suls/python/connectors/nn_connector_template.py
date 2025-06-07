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
ALPHABET_NAME = None # must be set


def load_nn_model(path_to_model: str):
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
  pass


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
  global ALPHABET_NAME

  alphabet_dir = os.path.split(path_to_model)
  path_to_alphabet = os.path.join(alphabet_dir, ALPHABET_NAME)

  alphabet = pk.load(open(path_to_alphabet, "rb"))

  assert(alphabet is not None)
  return list(alphabet)

def get_types():
  """
  In case that the model is being used to infer types of predictions (e.g. binary- 
  or multiclass-classification), this function returns a list of the possible predictions. 
  We need this so that flexfringe can map to the correct outputs.
  """
  pass

if __name__ == "__main__":
  raise Exception("This is not a standalone script.")