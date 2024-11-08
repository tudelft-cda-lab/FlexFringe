"""
Does what you think it does.
"""

import os
import argparse

ACCEPT = "TRUE"
REJECT = "FALSE"

LABEL_MAP = {
  ACCEPT: "A",
  REJECT: "R"
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("infile", type=str, help="Path to input file")
  parser.add_argument("outfile", type=str, help="Where to write output")
  args = parser.parse_args()
  
  inf_path = args.infile
  if not os.path.isfile(inf_path):
    raise ValueError("File does not exist: {}".format(inf_path))
  
  outf_path = args.outfile
  with open(inf_path, "r") as infh, open(outf_path, "w") as outfh:
    sequences = list()
    labels = list()
    alphabet = set()

    for line in infh:
      if len(line.strip()) == 0:
        continue
      
      line = line.strip().split()
      s = line[0]
      l = LABEL_MAP[line[1]]

      sequences.append([symbol for symbol in s])
      labels.append(l)
      for symbol in s:
        alphabet.add(symbol)
      
    outfh.write("{} {}\n".format(len(sequences), len(alphabet)))
    for line, label in zip(sequences, labels):
      outfh.write("{} {} {}\n".format(label, len(line), " ".join(line)))