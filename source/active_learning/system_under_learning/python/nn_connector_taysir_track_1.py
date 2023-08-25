"""
Connects to a sigmoid output network, i.e. a network with 
binary output. 

This file is used for MLFlow models, and we used it to reverse-engineer the 
DFAs from the TAYSIR competition.
"""

import mlflow

model = None
r_alphabet = None

def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  if model is not None:
    print("Model already loaded")
    return
  model = mlflow.pytorch.load_model(path_to_model)
  print("Model loaded successfully")

def map_sequence(seq: list):
  """Maps the sequences from the flexfringe input (integer values starting from zero
  strings, depending on implementation) to the input as needed by the network.

  Args:
      seq (list): The sequence as a list.

  Returns:
      list() or equivalent sequence type: The sequence mapped so that the network can take it.
  """
  global r_alphabet
  return [r_alphabet[symbol] for symbol in seq] # to compensate for mismatch in between flexfringe and network input


def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
  assert(model is not None)
  
  seq_mapped = map_sequence(seq)
  
  seq_one_hot = model.one_hot_encode(seq_mapped)
  y_pred = model.predict(seq_one_hot)
  
  return float(y_pred)


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
  global model
  global r_alphabet

  if model is None:
    load_nn_model(path_to_model)

  try: # RNNs
    nb_letters = model.input_size - 1
  except: # Transformer
    nb_letters = model.distilbert.config.vocab_size - 2

  r_alphabet = {str(i): i for i in range(nb_letters)}

  assert(r_alphabet is not None)
  return list(r_alphabet.keys())

if __name__ == "__main__":
  raise Exception("This script is not meant as a standalone.")