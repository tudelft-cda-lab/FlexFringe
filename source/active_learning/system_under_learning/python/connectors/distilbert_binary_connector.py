"""
This is a connector for the distilbert model from the transformer library.
"""

import os
import pickle as pk

import torch
import torch.nn.functional as F

# we need these variables globally, as they represent the status of our program
model = None 

DATASET_NAME = None # "dataset_problem_04.03.TLT.2.1.2.pk"
SOS = None
EOS = None
PAD = None # the padding symbol. Will be set by the get_alphabet() function
MAXLEN = None # max length of a sequence. Used for padding and set by get_alphabet() function

def make_dict(**kwargs):
    return kwargs

def get_forward_dict(x, y, mask, output_attentions=False):
    forward_dict = make_dict(
        input_ids=x, # the training data?
        labels=y, # the training labels
        attention_mask=mask, # TODO: we can do this to improve the models I suppose
        head_mask=None,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )
    return forward_dict

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

def construct_attn_mask(lengths, maxlen):
    """
    Lengths is a list. For each sequence in input_ids it gives the length
    """
    res = torch.ones((len(lengths), maxlen))
    for i, l in enumerate(lengths):
        res[i, l+1:] = 0
    return res

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  padding = [PAD] * (MAXLEN - len(seq))
  padded_seq = seq + padding
  padded_seq = torch.LongTensor(padded_seq).reshape(1, -1)
  mask = construct_attn_mask([len(seq)-1], MAXLEN)

  model_input = get_forward_dict(padded_seq, None, mask)
  with torch.no_grad():
    res = model(**model_input)
  res = F.softmax(res.logits, dim=1)
  res = torch.argmax(res, dim=1).numpy().tolist()

  return res[0]


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
  global DATASET_NAME
  global MAXLEN

  global SOS
  global EOS
  global PAD 

  model_id = path_to_model.split("_")[-1][:-3]
  DATASET_NAME = "dataset_problem_{}.pk".format(model_id)

  alphabet_dir = os.path.split(path_to_model)[:-1]
  alphabet_dir = "/".join(alphabet_dir)
  path_to_dataset = os.path.join(alphabet_dir, DATASET_NAME)

  dataset_dict = pk.load(open(path_to_dataset, "rb"))

  alp_size = dataset_dict["alphabet_size"]
  symbol_dict = dataset_dict["symbol_dict"]

  SOS = dataset_dict["SOS"]
  EOS = dataset_dict["EOS"]
  PAD = dataset_dict["PAD"]
  MAXLEN = dataset_dict["maxlen"]

  alphabet = [int(v) for k, v in symbol_dict.items() if not k=="<SOS>" and not k=="<EOS>"]
  assert(alphabet is not None)
  print("Original alphabet dict: ", symbol_dict, "\nThe alphabet to give to flexfringe: ", alphabet)
  return alphabet


if __name__ == "__main__":
  #load_nn_model("model_04.03.TLT.2.1.2.pk")
  #_ = get_alphabet(".")
  #_ = do_query([4, 0, 5])
  raise Exception("This is not a standalone script.")