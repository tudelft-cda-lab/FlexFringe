"""
Connects to a sigmoid output network, i.e. a network with 
binary output. 

This file is used for MLFlow models, and we used it to reverse-engineer the 
DFAs from the TAYSIR competition.
"""

import mlflow

model = None

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

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
    
  seq_one_hot = model.one_hot_encode(seq)
  y_pred = model.predict(seq_one_hot)
  
  return float(y_pred)


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
  global model
  global alphabet

  if model is None:
    load_nn_model(path_to_model)

  try: # RNNs
    nb_letters = model.input_size - 1
  except: # Transformer
    nb_letters = model.distilbert.config.vocab_size - 2

  return list(range(nb_letters))

if __name__ == "__main__":
  raise Exception("This script is not meant as a standalone.")