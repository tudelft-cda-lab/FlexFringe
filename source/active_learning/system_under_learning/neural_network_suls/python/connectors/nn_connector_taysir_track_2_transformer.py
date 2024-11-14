"""
Connects to a sigmoid output network, i.e. a network with 
binary output. 

This file is used for MLFlow models, and we used it to reverse-engineer the 
DFAs from the TAYSIR competition.
"""

import torch
import mlflow
import numpy

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


def make_future_masks(words:torch.Tensor):
    masks = (words != 0)
    b,l = masks.size()
    #x = einops.einsum(masks, masks, "b i, b j -> b i j")
    x = torch.einsum("bi,bj->bij",masks,masks)
    x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril()
    x += torch.eye(l,dtype=torch.bool, device=x.device)
    return x.type(torch.int8)

def predict_next_symbols(model, word):
    """
    Args:
        whole word (list): a complete sequence as a list of integers
    Returns:
        the predicted probabilities of the next ids for all prefixes (2-D ndarray)
    """
    word = [ [ a+1 for a in word ] ]
    word = torch.IntTensor(word)
    model.eval()
    with torch.no_grad():
        attention_mask = make_future_masks(word)
        out = model.forward(word, attention_mask=attention_mask)
        out = torch.nn.functional.softmax(out.logits[0], dim=1)
        return out.detach().numpy()[:, 1:] #  the probabilities for padding id (0) are removed

def predict_transformer(model, word):
    probs = predict_next_symbols(model, word[:-1])
    probas_for_word = [probs[i,a] for i,a in enumerate(word[1:])]
    value = numpy.array(probas_for_word).prod()
    return float(value)

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