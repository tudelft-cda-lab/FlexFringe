import os

#output_file = "/home/robert/Documents/code/FlexFringe/data/PAutomaC-competition_sets/1.pautomac.test.2.dat"



for scenario in range(1, 48+1):
  input_file = "/home/robert/Documents/code/FlexFringe/data/PAutomaC-competition_sets/{}.pautomac.test.dat".format(scenario)
  output_file = "/home/robert/Documents/code/FlexFringe/data/PAutomaC-competition_sets/{}.pautomac.test.2.dat".format(scenario)
  outf = open(output_file, "wt")

  for i, line in enumerate(open(input_file)):
      outf.write(line)
  outf.close()

for scenario in range(1, 48+1):
  input_file = "/home/robert/Documents/code/FlexFringe/data/PAutomaC-competition_sets/{}.pautomac.test.dat".format(scenario)
  output_file = "/home/robert/Documents/code/FlexFringe/data/PAutomaC-competition_sets/{}.pautomac.test.2.dat".format(scenario)

  os.remove(input_file)
  os.rename(output_file, input_file)