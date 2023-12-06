"""
Example for the weighted output model(s) trained.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle as pk

import numpy as np
import tensorflow.keras
from tensorflow.keras.models import load_model

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
  model = load_model(path_to_model)
  print("Model loaded successfully")


def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  global model

  array = np.array(seq)
  #print("Before prediction: {}".format(seq))
  #res = model(array.reshape(1, -1), training=False)
  res = model.predict(array.reshape(1, -1), verbose=0)
  #res = model(np.array(seq).reshape(1, -1), train=False)

  #print("After prediction")
  #res = res.numpy().reshape(-1).tolist()
  res = res.reshape(-1).tolist()
  #print("Predicted")

  return res


def get_alphabet(path_to_model_file: str):
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must a list of int objects, representing the possible inputs of the network. 
  Flexfringe will take care of the internals.

  Args:
      path_to_model_file (str): What you think it is.

  Returns:
      alphabet: list(int): The possible inputs the network can take (including <SOS> and 
      <EOS>).
  """
  path_split = path_to_model_file.split('/')
  alph_path = os.path.join("/".join(path_split[:-1]), "test_data.pk")

  print("Trying to load alphabet data from location {}".format(alph_path))
  a_data = pk.load(open(alph_path, "rb"))
  print("Successfully loaded the alphabet data.")

  event_mapping = a_data["event_mapping"]
  print("The events are mapped as follows:\n\n{}".format(event_mapping))

  res = list(event_mapping.values())
  return res


if __name__ == "__main__":
  #raise Exception("This script is not meant as a standalone.")
  load_nn_model("model.keras")
  for i in range(200):
    do_query([79, 1, 2])