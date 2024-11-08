"""
This is a connector for the transformer. Apart from the query result it also returns an internal representation,
which can be the attention matrix, the attention output etc.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 8.1.24.
"""

import os
import pickle as pk

import torch

# we need these variables globally, as they represent the status of our program
model = None 

DATASET_NAME = "dataset_problem_04.03.TLT.2.1.2.pk"
SOS = None
EOS = None
PAD = None # the padding symbol. Will be set by the get_alphabet() function
MAXLEN = None # max length of a sequence. Used for padding and set by get_alphabet() function


def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  global activations

  model = torch.load(path_to_model)
  for param in model.parameters():
      param.requires_grad = False

  print("Model loaded successfully")


def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  padding = [PAD] * (MAXLEN - len(seq))
  padded_seq = seq + padding

  query_string = torch.reshape(torch.tensor(padded_seq), (-1, 1))
  with torch.no_grad():
    res = model(query_string) # last_index=len(seq)-1
  res = torch.squeeze(res).numpy().tolist()

  last_symbol_index = len(seq)-1
  print("seq: ", padded_seq)
  print("index: ", last_symbol_index)
  
  return res[last_symbol_index]


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
  #global DATASET_NAME
  global MAXLEN

  global SOS
  global EOS
  global PAD 

  alphabet_dir = os.path.split(path_to_model)[:-1]
  alphabet_dir = "/".join(alphabet_dir)
  path_to_dataset = os.path.join(alphabet_dir, DATASET_NAME)

  dataset_dict = pk.load(open(path_to_dataset, "rb"))
  print("dataset dict: ", dataset_dict)

  alp_size = dataset_dict["alphabet_size"]
  symbol_dict = dataset_dict["symbol_dict"]

  SOS = dataset_dict["SOS"]
  EOS = dataset_dict["EOS"]
  PAD = dataset_dict["PAD"]
  MAXLEN = dataset_dict["maxlen"]

  alphabet = [int(v) for k, v in symbol_dict.items() if not k=="<SOS>" and not k=="<EOS>"]
  assert(alphabet is not None)
  print("The alphabet: ", alphabet)
  return alphabet


if __name__ == "__main__":
  raise Exception("This is not a standalone script.")