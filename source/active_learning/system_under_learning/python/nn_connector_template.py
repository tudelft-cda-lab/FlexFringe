"""
This file is a template to connect the SUL to 
a neural network and is meant to capture and transfer
outputs and queries in between the two systems.

Created by Robert Baumgartner, 31.7.23.
"""

# we need these variables globally, as they represent the status of our program
MODEL_NAME = ""
model = None 

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

if __name__ == "__main__":
  raise Exception("This is not a standalone script.")