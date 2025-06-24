import os
import random
from collections import defaultdict
from typing import List, Tuple

DATA_DIR = "../data/staminadata"
OUTPUT_DIR = "../data/stamina_split_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_dataset(lines: List[str]):
    num_traces, alphabet_string = lines[0].strip().split()
    num_traces = int(num_traces)
    alphabet_size = int(alphabet_string.split(':')[0])
    traces = []
    for line in lines[1:]:
        parts = line.strip().split()
        label = int(parts[0])
        length = int(parts[1])
        trace = parts[2:]
        assert length == len(trace), f"Invalid length in trace: {line}"
        traces.append((label, trace))
    return num_traces, alphabet_size, traces

def make_unique(traces: List[Tuple[int, List[str]]]):
    unique = {}
    for label, trace in traces:
        key = (label, tuple(trace))
        unique[key] = (label, trace)
    return list(unique.values())

def train_test_split(traces, test_ratio = 0.2):
    pos = [t for t in traces if t[0] == 1]
    neg = [t for t in traces if t[0] == 0]
    pos_split = int(len(pos) * (1 - test_ratio))
    neg_split = int(len(neg) * (1 - test_ratio))
    train = pos[:pos_split] + neg[:neg_split]
    test = pos[pos_split:] + neg[neg_split:]
    return train, test

def split(traces, valid_ratio = 0.15, test_ratio = 0.15):
    pos = [t for t in traces if t[0] == 1]
    neg = [t for t in traces if t[0] == 0]
    pos_split_1 = int(len(pos) * (1 - valid_ratio - test_ratio))
    neg_split_1 = int(len(neg) * (1 - valid_ratio - test_ratio))
    pos_split_2 = int(len(pos) * (1 - test_ratio))
    neg_split_2 = int(len(neg) * (1 - test_ratio))
    train = pos[:pos_split_1] + neg[:neg_split_1]
    valid = pos[pos_split_1:pos_split_2] + neg[neg_split_1:neg_split_2]
    test = pos[pos_split_2:] + neg[neg_split_2:]
    return train, valid, test

def format_trace(label, trace):
    return f"{label} {len(trace)} {' '.join(trace)}"

def write_dataset(path, alphabet_size, traces):
    with open(path, 'w') as f:
        f.write(f"{len(traces)} {alphabet_size}\n")
        for label, trace in traces:
            f.write(format_trace(label, trace) + "\n")
            

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".dat"):
        input_path = os.path.join(DATA_DIR, filename)
        print(f"Processing file: {input_path}")

        with open(input_path) as f:
            lines = f.readlines()

            num_traces, alphabet_size, traces = parse_dataset(lines)
            unique_traces = make_unique(traces)
            train, valid, test = split(unique_traces, 0.15, 0.15)

            base_name = os.path.splitext(filename)[0]
            train_path = os.path.join(OUTPUT_DIR, f"{base_name}_train.txt.dat")
            valid_path = os.path.join(OUTPUT_DIR, f"{base_name}_valid.txt.dat")
            test_path = os.path.join(OUTPUT_DIR, f"{base_name}_test.txt.dat")

            write_dataset(train_path, alphabet_size, train)
            write_dataset(valid_path, alphabet_size, valid)
            write_dataset(test_path, alphabet_size, test)

            pos = 0
            for label, trace in train:
                if label == 1:
                    pos += 1
            print("Train sparsity: ", pos / len(train))
            pos = 0
            for label, trace in valid:
                if label == 1:
                    pos += 1
            print("Validation sparsity: ", pos / len(valid))
            pos = 0
            for label, trace in test:
                if label == 1:
                    pos += 1
            print("Test sparsity: ", pos / len(test))

    # print(f"Original traces: {num_traces}")
    # print(f"Unique traces: {len(unique_traces)}")
    # print(f"Train set size: {len(train)}")
    # print(f"Test set size: {len(test)}")