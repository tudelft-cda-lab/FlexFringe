"""
Connects to a sigmoid output network, i.e. a network with 
binary output. 

This file is used for MLFlow models, and we used it to reverse-engineer the 
DFAs from the TAYSIR competition.
"""

import torch
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

def predict_transformer(model, word):
    """
    Note: In this function, each id in the word is added to 2 before being input to the model,
    since ids 0 and 1 are used as special tokens.
        0 : padding id
        1 : classification token id
    Args:
        word: list of integers 
    """
    word = [ [1] + [ a+2 for a in word ] ]
    word = torch.IntTensor(word)
    with torch.no_grad():
        out = model(word)
        return (out.logits.argmax().item())

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
    
  return float(predict_transformer(model, seq))


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

  if model is None:
    load_nn_model(path_to_model)

  nb_letters = model.distilbert.config.vocab_size - 2

  return list(range(nb_letters))


if __name__ == "__main__":
  raise Exception("This script is not meant as a standalone.")