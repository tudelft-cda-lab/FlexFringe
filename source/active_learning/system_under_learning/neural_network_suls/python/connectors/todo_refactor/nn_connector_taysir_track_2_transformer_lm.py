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

RANGE = None
SOS_SYMBOL = None
EOS_SYMBOL = None

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
    print("einsum: ", x)
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
    #word = [ [ a+1 for a in word ] ]
    word = [ [ a for a in word ] ]
    word = torch.IntTensor(word)
    model.eval()
    with torch.no_grad():
        attention_mask = make_future_masks(word)
        #print(attention_mask, list(attention_mask.size()))
        out = model.forward(word, attention_mask=attention_mask)
        out = torch.nn.functional.softmax(out.logits[0], dim=1)
        #return out.detach().numpy()[:, 1:] #  the probabilities for padding id (0) are removed
        return out.detach().numpy()

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  seq.append(EOS_SYMBOL)
  y_pred = predict_next_symbols(model, seq)
  # the transformer has been initialized on more symbols that it is used to, hence RANGE+1. Also, index 0 is padding symbol
  return list(y_pred[-1, 1:RANGE]) 

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
  global RANGE
  global SOS_SYMBOL, EOS_SYMBOL

  if model is None:
    load_nn_model(path_to_model)
  nb_letters = model.distilbert.config.vocab_size

  RANGE = nb_letters - 1 # last index used
  print("nb_letters: ", nb_letters)

  # nb_letters includes SOS and EOS as the last two symbols in this dataset, and we have to start counting from 0
  SOS_SYMBOL = nb_letters - 2
  EOS_SYMBOL = nb_letters - 1
  print("Alphabet for TAYSIR model initialized. <SOS>: {}, <EOS>: {}".format(SOS_SYMBOL, EOS_SYMBOL))

  return [str(x) for x in range(1, nb_letters)] # 0 is padding symbol

def get_sos_eos():
  """
  Returns the SOS and EOS symbols of the model as strings as a list. SOS is at 
  position 0, EOS is at position 1.

  Potential side effect: If you use the SOS and EOS symbols before and after querying the model, 
  this would be a good chance to initialize them here.
  """
  

  if model is None:
    print("Error in Python script in get_sos_eos(): Model not initialized.")
    raise RuntimeError()

  nb_letters = model.distilbert.config.vocab_size



  return [SOS_SYMBOL, EOS_SYMBOL]

if __name__ == "__main__":
  path = "2.10.taysir.model"
  load_nn_model(path)
  get_alphabet(path)
  get_sos_eos()
  do_query([])
  raise Exception("This script is not meant as a standalone.")