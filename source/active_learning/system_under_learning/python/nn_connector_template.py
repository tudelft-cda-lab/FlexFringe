"""
This file is a template to connect the SUL to 
a neural network and is meant to capture and transfer
outputs and queries in between the two systems.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 31.7.23.
"""

# we need these variables globally, as they represent the status of our program
MODEL_NAME = ""
model = None 

alphabet = None

def load_model():
  """Load a model and writes it into the global model-variable
  """
  global model
  global MODEL_NAME
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

def get_alphabet():
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must be of dict()-type, mapping input alphabet of type str() to internal 
  representation in integer values.

  Side effect (must be implemented!): If the alphabet is uninitialized (None), set the alphabet accordingly.

  alphabet: dict() [str : int]
  """
  global alphabet
  pass

if __name__ == "__main__":
  raise Exception("This is not a standalone script.")