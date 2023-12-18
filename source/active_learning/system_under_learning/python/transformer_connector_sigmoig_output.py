"""
Connects to a sigmoid output network, i.e. a network with 
binary output. 

This file is used for MLFlow models, and we used it to reverse-engineer the 
DFAs from the TAYSIR competition.
"""
import os
import pickle as pk

import torch

model = None


def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model

  model = torch.load(path_to_model)
  print("Model loaded successfully")


def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model
  res = model.predict(torch.tensor(seq))
  return float(res.numpy())


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
  path = path_to_model.split('/')
  path_to_alphabet = os.path.join("/".join(path[:-1]))
  path_to_alphabet = os.path.join(path_to_alphabet, "dataset.pk")

  data = pk.load(open(path_to_alphabet, "rb"))
  print("Alphabet size: ", data["alphabet_size"])
  print("Symbol mapping: ", data["symbol_dict"])
  print("Label mapping: ", data["label_dict"])

  return list(range(data["alphabet_size"]))


if __name__ == "__main__":
  raise Exception("This script is not meant as a standalone.")