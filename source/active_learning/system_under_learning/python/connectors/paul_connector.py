"""
Like the binary state causal transformer connector, but also returns confidence of transformer as well.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 10.9.24.
"""

import os
import pickle as pk

import re

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import softmax

import transformers

import sys
#sys.path.append("/home/robert/Documents/code/Flexfringe/source/active_learning/system_under_learning/python/util")
sys.path.append("../util")
from distillbert_for_language_model import DistilBertForTokenClassification
#from transformers import DistilBertForTokenClassification

DEVICE = "cpu" # we do not do batch processing, so CPU should be good enough

# we need these variables globally, as they represent the status of our program
model = None 

# the following will be set by get_alphabet()
SOS = None
EOS = None
PAD = None 

MAXLEN = None # max length of a sequence. Used for padding and set by get_alphabet() function
ALPHABET_SIZE = None

def make_dict(**kwargs):
    return kwargs

def init_global_variables(path_to_model: str):
  global MAXLEN, ALPHABET_SIZE, SOS, EOS, PAD

  path_split = os.path.split(path_to_model)
  data_dir = os.path.join(*path_split[:-1])

  regex_pattern = "\d\d\.\d\d\.[A-Z]{2,5}\.\d\.\d\.\d"
  problem_id = re.search(regex_pattern, path_to_model).group(0)
  print("Identified problem {}".format(problem_id))

  path_to_dataset = os.path.join(data_dir, "dataset_problem_{}.pk".format(problem_id))
  dataset_dict = pk.load(open(path_to_dataset, "rb"))

  ALPHABET_SIZE = dataset_dict["alphabet_size"]
  SOS = dataset_dict["SOS"]
  EOS = dataset_dict["EOS"]
  PAD = dataset_dict["PAD"]
  MAXLEN = dataset_dict["maxlen"]

def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model

  init_global_variables(path_to_model)

  init_dict = make_dict( # WARNING: must be same as in training file
      # vocab_size=train_dataset.alphabet_size+3,
      vocab_size=ALPHABET_SIZE+5,
      num_labels=ALPHABET_SIZE+5,
      max_position_embeddings=MAXLEN+2,
      sinusoidal_pos_embds=True,
      use_pos_embds=True,
      
      n_layers=4,
      n_heads=12,
      # dim=train_dataset.alphabet_size*2,
      dim = 96,
      hidden_dim = 96,
      # hidden_dim=train_dataset.alphabet_size*2,
      activation="gelu",
      dropout=0.1,
      attention_dropout=0.1,
      seq_classif_dropout=0.2,
      PAD_TOKEN_id=PAD
  )

  model = DistilBertForTokenClassification(transformers.DistilBertConfig(**init_dict))
  print("Load state dict")
  model.load_state_dict(torch.load(path_to_model, map_location=torch.device(DEVICE)))
  print("Params")
  for param in model.parameters():
      param.requires_grad = False
  print("Model successfully loaded")

def make_tensor_causal_masks(words:torch.Tensor):
    masks = (words != PAD) # 1 : pass, 0 : blocked
    b,l = masks.size()
    x = torch.einsum("bi,bj->bij",masks,masks)
    # Mask with lower triangle including diagonal -> causal mask
    x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril() 
    x += torch.eye(l,dtype=torch.bool, device=x.device)
    return x.type(torch.int8)

def get_representation(output, last_token_idx):
  """Gets the attention. Make sure to keep the convention: Keys go from 1...number of attention vectors, 
  as 0 is reserved for the networks output.

  Returns:
      output (dict): 1...n_attn -> attn_vector
  """
  DO_FIRST_ONLY = True

  if DO_FIRST_ONLY:
    attn = torch.squeeze(output["hidden_states"][1].detach()).numpy() # (b_size, maxlen_seq, hidden_dim); b_size here will always be 1 and squeezed out!
    #attn = torch.squeeze(output["attentions"][0].detach()).numpy() # (b_size, n_heads, maxlen_seq, maxlen_seq); b_size here will always be 1 and squeezed out!
    #attn = np.mean(attn, axis=0) # using the attn and not the states
  elif False:
    pass # placeholder for different strategy
  else: # concatenate all of them
    attn = output.hidden_states[1]
    for i in range(2, len(output.hidden_states)):
      attn = torch.cat((attn, output.hidden_states[i]), dim=-1)    
    attn = torch.squeeze(attn.detach()).numpy()
  
  res = list()
  for i in range(last_token_idx+1):
    res.extend(list(attn[i]))

  return res

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Must return a dictionary. For efficiency the following convention holds for the 
  keys: 1 = prediction, 2 = attention

  Args:
      seq (list): List of ints.
  """
  padding = [PAD] * (MAXLEN - len(seq) - 1)
  padded_seq = seq + padding
  last_token_idx = len(seq) - 1

  query_string = torch.reshape(torch.tensor(padded_seq), (1, -1))
  mask = make_tensor_causal_masks(query_string)

  with torch.no_grad():
    model.eval()
    output = model(input_ids=query_string, attention_mask=mask, return_dict=True, output_attentions=False, output_hidden_states=True)
  logits = torch.squeeze(output.logits)

  preds = torch.argmax(logits[last_token_idx])
  confidence = torch.max(softmax(logits[last_token_idx]))
  if(logits.size(-1) > PAD + 1):
      preds = preds - PAD - 1
  if preds < 0 or preds > 1:
     print("Erroneous prediction: ", preds) # what now?
  
  representations = get_representation(output, last_token_idx)
  embedding_dim = int(len(representations) / len(seq))

  res = [confidence.item(), preds.item(), embedding_dim]
  res.extend(representations)

  return res


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
  path_split = os.path.split(path_to_model)
  data_dir = os.path.join(*path_split[:-1])
  
  regex_pattern = "\d\d\.\d\d\.[A-Z]{2,5}\.\d\.\d\.\d"
  problem_id = re.search(regex_pattern, path_to_model).group(0)

  path_to_dataset = os.path.join(data_dir, "dataset_problem_{}.pk".format(problem_id))
  dataset_dict = pk.load(open(path_to_dataset, "rb"))
  symbol_dict = dataset_dict["symbol_dict"]

  print(dataset_dict)

  alphabet = [int(v) for k, v in symbol_dict.items() if not k=="<SOS>" and not k=="<EOS>"]
  assert(alphabet is not None)
  print("The alphabet: ", alphabet)
  return alphabet


def get_hidden_representation(res: list):
  """
  Only for debugging purposes. We want to manually check our output.

  res: The result as returned by do_query.
  """
  pred = res[0]
  embed_dim = res[1]
  n_hidden = (len(res) - 2) / embed_dim
  n_hidden = int(n_hidden)

  reps = list()
  for i in range(n_hidden):
    start = 2 + i * embed_dim
    reps.append([res[j] for j in range(start, start+embed_dim)])
    print(len(reps[-1]))
  
  return reps
    

if __name__ == "__main__":
  # {'c': 0, 'b': 1, 'd': 2, 'a': 3}
  model_path = "/home/robert/Documents/code/Flexfringe/data/active_learning/mlregtest/trained_models/distilbert_problem_04.04.SL.2.1.0_mid.pk.finetuned"
  get_alphabet(model_path)
  load_nn_model(model_path)
  #seq = [4, 1, 1, 5]
  seq = [4, 1, 3, 3, 1, 5]
  res = do_query(seq)
  print(type(res), len(res))
  print("here come the hidden reps: ")
  hreps = get_hidden_representation(res)
  for i in range(len(hreps)):
     print("Index: {}, char: {}, hreps: {}\n\n\n".format(i, seq[i], hreps[i][:3]))

  raise Exception("This is not a standalone script.")