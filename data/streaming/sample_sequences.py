"""
Because the test-sequence file is really large, and we want to save time evaluating, we will sample here.
"""
import random

infile_name = "test_set_clean.dat"
label_filename = "labels_test_set_clean.txt"

outfile_name = "sequences_sampled.dat"
label_outfile_name = "labels_sampled.txt"

label_counter = dict()
samples_per_label = 100000

n_sequences = 0
alphabet_size = 0

def get_label(labels: set):
    if len(labels) == 1 and "0" in labels:
        return "0"
    return "1"

label_file = open(label_filename, "rt")
for i, line in enumerate(open(infile_name, "rt")):
    if i == 0:
        alphabet_size = line.split()[-1]

    labels = set(label_file.readline().split())
    label = get_label(labels)
    if not label in label_counter:
        label_counter[label] = 0

    label_counter[label] += 1
label_file.close()
print("Done counting: ", label_counter)

probabilities = {label: samples_per_label/float(label_counter[label]) for label in label_counter.keys()}
print("The probabilites are ", probabilities)

label_file = open(label_filename, "rt")
label_outfile = open(label_outfile_name, "wt")
# TODO: write header here
outf = open(outfile_name, "wt")
for i, line in enumerate(open(infile_name, "rt")):
    if i == 0:
        continue
    
    label_line = label_file.readline()
    labels = set(label_line)
    label = get_label(labels)
    
    p = random.random()
    if p < probabilities[label]:
        outf.write(line)
        label_outfile.write(label_line)
label_file.close()
label_outfile.close()
outf.close()