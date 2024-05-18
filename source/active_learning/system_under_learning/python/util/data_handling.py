"""
Custom dataset for handling the transformer data we generate. 

TODO: Cannot yet react to symbols we see in the test set, but not 
in the train set (see workflow of experiments for more details)
"""

import os
import pickle as pk

import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, datapath: str, maxlen: int=None, pad_sequences: bool=True, max_sequences: int=None):
        """Inits the model. 

        Args:
            datapath (str): The path to the abbadingo formatted input. 
            maxlen (int, optional): The maximum expected length from the input without <EOS> and <SOS>. Sequences shorther than that will be padded with a special PAD symbol. Defaults to None.
            pad_sequences (bool, optional): Padding, yes or no? Defaults to True.
            max_sequences (int, optional): Maximum number of sequences we want to process. Helps with limited memory resources. Defaults to None.
        """
        super().__init__()
        
        assert(os.path.isfile(datapath))
        self.symbol_dict = dict()
        self.label_dict = dict()
        self.sequences, self.labels, self.sequence_lengths = self._read_sequences(datapath, max_sequences)
        print("Sequences loaded. Some examples: \n{}".format(self.sequences[:3]))
        
        self.SOS = self.alphabet_size
        self.EOS = self.alphabet_size + 1
        self.PAD = self.alphabet_size + 2
        self.maxlen = maxlen + 2  # +2 for EOS/PAD and SOS 
        self.pad_sequences = pad_sequences
        
    def encode_sequences(self):
        self.ordinal_seq, self.ordinal_seq_sr = self._ordinal_encode_sequences(self.sequences)
        self.one_hot_seq, self.one_hot_seq_sr = self._one_hot_encode_sequences(self.sequences)
        
        del self.sequences
        self.sequences = None
        
        print("The symbol dictionary: {}".format(self.symbol_dict))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ordinal_seq[idx], self.ordinal_seq_sr[idx], self.one_hot_seq[idx], \
               self.one_hot_seq_sr[idx], self.labels[idx], self.sequence_lengths[idx]
       
    def _read_sequences(self, datapath: str, max_sequences: int):
        sequences = list()
        labels = list()
        sequence_lengths = list()
        
        for i, line in enumerate(open(datapath)):
            if i == 0:
                line = line.split()
                self.alphabet_size = int(line[1])
                print("Alphabet size: ", self.alphabet_size)
                continue
            elif max_sequences and i-1 >= max_sequences:
                break
            
            line = line.split()
            label = line[0]
            if not label in self.label_dict:
                self.label_dict[label] = len(self.label_dict)
            label = self.label_dict[label]
            labels.append(label)
            
            sequences.append(line[2:])
            sequence_lengths.append(len(line) - 1)
        return sequences, labels, sequence_lengths
    
    def _pad_one_hot(self, sequences: list, do_eos: bool=False):
        for i in range(len(sequences)):
            seq = sequences[i]
            #print("Before one hot:\n{}".format(seq))
            current_size = len(seq)
            
            t = torch.zeros((self.maxlen - current_size, self.alphabet_size + 3), dtype=torch.float32)
            t[:, self.PAD] = 1
            if do_eos and self.maxlen > current_size:
                t[0, self.PAD] = 0
                t[0, self.EOS] = 1
            
            seq = torch.cat((seq, t), dim=0)
            sequences[i] = seq
            #print("After one hot:\n{}".format(seq))
        return sequences
    
    def _one_hot_encode_sequences(self, strings: list):
        res = list()
        res_sr = list()
        for string in strings:
            x1, x2 = self._one_hot_encode_string(string)
            res.append(x1)
            res_sr.append(x2)
            
        if self.pad_sequences:
            res = self._pad_one_hot(res)
            res_sr = self._pad_one_hot(res_sr)

        return res, res_sr
    
    def _one_hot_encode_string(self, string: list):
        encoded_string = torch.zeros((len(string)+2, self.alphabet_size + 3), dtype=torch.float32) # alphabet_size + 3 because SOS, EOS, padding token
        encoded_string[0][self.SOS] = 1
        encoded_string[-1][self.EOS] = 1

        encoded_string_sl = torch.zeros((len(string)+2, self.alphabet_size + 3), dtype=torch.float32)
        encoded_string_sl[-2][self.EOS] = 1
        encoded_string_sl[-1][self.PAD] = 1

        for i, symbol in enumerate(string):
            if not symbol in self.symbol_dict:
                self.symbol_dict[symbol] = len(self.symbol_dict)

            encoded_string[i+1][self.symbol_dict[symbol]] = 1
            encoded_string_sl[i][self.symbol_dict[symbol]] = 1
        encoded_string_sl.requires_grad_()
        return encoded_string, encoded_string_sl
    
    def _pad_ordinal(self, sequences: list, do_eos: bool=False):
        for i in range(len(sequences)):
            seq = sequences[i]
            #print("Before ordinal:{}".format(seq))
            current_size = len(seq)
            
            t = torch.ones((self.maxlen - current_size,), dtype=torch.long)
            t = t*self.PAD 
            if do_eos and self.maxlen > current_size:
                t[0] = self.EOS
            
            seq = torch.cat((seq, t), dim=0)
            sequences[i] = seq
            #print("After ordinal:{}".format(seq))
        return sequences
    
    def _ordinal_encode_sequences(self, strings: list):
        res = list()
        res_sr = list()
        for string in strings:
            x1, x2 = self._ordinal_encode_string(string)
            res.append(x1)
            res_sr.append(x2)
        
        if self.pad_sequences: 
            res = self._pad_ordinal(res)
            res_sr = self._pad_ordinal(res_sr)
        return res, res_sr
    
    def _ordinal_encode_string(self, string: list):
        encoded_string = torch.zeros((len(string)+2,), dtype=torch.long)
        encoded_string[0] = self.SOS
        encoded_string[-1] = self.EOS

        encoded_string_sl = torch.zeros((len(string)+2,), dtype=torch.long)
        encoded_string_sl[-2] = self.EOS
        encoded_string_sl[-1] = self.PAD

        for i, symbol in enumerate(string):
            if not symbol in self.symbol_dict:
                self.symbol_dict[symbol] = len(self.symbol_dict)

            encoded_string[i+1] = self.symbol_dict[symbol]
            encoded_string_sl[i] = self.symbol_dict[symbol]
        return encoded_string, encoded_string_sl
    
    def get_alphabet_size(self):
        return self.alphabet_size
    
    def initialize(self, path: str="dataset.pk"):
        """Initializes the dataset with the data-dict saved in path.
        Overrides the following attributes:
        SOS, EOS, PAD, maxlen, alphabet_size, symbol_dict (i.e. the alphabet
        mapping), and label_dict.

        Must be called before encoding the sequences if you want it to take
        effect. In case any of the attributes does not fit from the loading process,
        e.g. the maxlen, please set it manually.

        Args:
            path (str, optional): Path to the dataset dict. Defaults to "dataset.pk".
        """
        data = pk.load(open(path, "rb"))
        self.alphabet_size = data["alphabet_size"]
        self.symbol_dict = data["symbol_dict"]
        self.label_dict = data["label_dict"]

        self.maxlen = data["maxlen"]
        self.SOS = data["SOS"]
        self.EOS = data["EOS"]
        self.PAD = data["PAD"]
        
    def save_state(self, path: str="dataset.pk"):
        """Saves the state of the dataset in a dataset dictionary containing
        all the metadata used to encode the sequences into oridinal and 
        one-hot sequences.

        Args:
            path (str, optional): The path to save at. Defaults to "dataset.pk".
        """
        data = dict()
        data["alphabet_size"] = self.alphabet_size
        data["symbol_dict"] = self.symbol_dict
        data["label_dict"] = self.label_dict
        data["maxlen"] = self.maxlen

        data["SOS"] = self.SOS
        data["EOS"] = self.EOS
        data["PAD"] = self.PAD

        pk.dump(data, open(path, "wb"))


class DistilBertData(Dataset):
    def __init__(self, data, labels, attn_mask):
        self.data = data
        self.labels = torch.zeros((len(labels), 2))
        for i, label in enumerate(labels):
            self.labels[i, label] = 1
        self.attn_mask = attn_mask

    def __len__(self):
        return list(self.data.size())[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.attn_mask[idx]