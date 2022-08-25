import sys

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

def get_errors(filepath):
  """[Gets the errors out of one perplexity file]
  """
  values = []
  with open(filepath, 'r') as f:
    for line in f:
      line = line.split()
      val = line[-1]
      if val == "zero":
        continue
      values.append(float(val))
  
  return values


if __name__ == "__main__":
  arguments = sys.argv
  graphs = []
  for i in range(1, len(arguments)):
    filepath = arguments[i]
    graphs.append((get_errors(filepath), filepath))

  fig = plt.figure(figsize=(20, 12))
  for i, graph_pair in enumerate(graphs):
    graph = graph_pair[0]
    name = graph_pair[1]

    #if "cms" in name:
    #  name = "Sketches"
    #elif "alergia" in name:
    #  name = "Alergia"
    #if i == 0:
    #  name = "Alergia ktail == 0"
    #elif i == 1:
    #  name = "Alergia ktail == 1"
    #if i == 2:
    #  name = "Alergia with ktail == 3"
    #elif i == 3:
    #  name = "Alergia with ktail == 3"

    plt.plot(range(1, len(graph) + 1), graph, marker="o", label=name, markersize=10)

  plt.xlabel("Pautomac-scenario")
  plt.ylabel("Perplexity error")
  plt.legend()

  plt.savefig("errors.jpg")