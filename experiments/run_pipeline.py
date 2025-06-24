from pipeline import *
import os
import shutil
import matplotlib.pyplot as plt
from itertools import combinations
import sys
import numpy as np
from scipy.stats import ttest_rel

DATA_DIR = "../data/stamina_split_validation"
RESULTS_DIR = "../data/stamina_ensembles"
ENSEMBLE_DIRS = ["ensemble_0.1", "ensemble_0.4", "ensemble_0.7", "ensemble_boost"]
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_index(filename):
    return int(filename.split("_")[0])

def process_dataset(data_dir):
    accuracies = []
    for filename in sorted(os.listdir(data_dir), key=extract_index):
        if filename.endswith("_train.txt.dat"):
            base_name = os.path.join(data_dir, filename).replace("_train.txt.dat", "")
            train = base_name + "_train.txt.dat"
            test = base_name + "_test.txt.dat"
            apta_file = train + ".ff.final.json"

            learn_single_dfa(train)
            acc = predict_single_dfa(test, apta_file)
            accuracies.append(acc)
        for file in os.listdir(data_dir):
                if file.endswith(".json") or file.endswith(".dot") or file.endswith(".result"):
                    os.remove(os.path.join(data_dir, file))
    return accuracies

def compute_train_density(train_path):
    with open(train_path) as f:
        lines = f.readlines()
        pos = sum(1 for line in lines[1:] if line.startswith("1 "))
        total = len(lines) - 1
        return pos / total if total else 0
    
def get_train_file_path(dataset_number):
    for file in os.listdir(DATA_DIR):
        if file.startswith(f"{dataset_number}_") and file.endswith("_train.txt.dat"):
            return os.path.join(DATA_DIR, file)
    return None

def run_random():
    for filename in sorted(os.listdir(DATA_DIR), key=extract_index):
        if filename.endswith("_train.txt.dat"):
            full_path = os.path.join(DATA_DIR, filename)
            base_name = full_path.replace("_train.txt.dat", "")
            dataset_number = base_name.split("\\")[-1].split("_")[0]

            train = base_name + "_train.txt.dat"
            valid = base_name + "_valid.txt.dat"
            test = base_name + "_test.txt.dat"
            ensemble_file = train + ".ff.final_ensemble.json"
            apta_file = train + ".ff.final.json"
            print("Base name: ", base_name)
            print("Train: ", train)
            print("Test: ", test)
            print("Ensemble file: ", ensemble_file)
            print("Apta file: ", apta_file)

            ensemble_random(train, 10, 0.999, 1)
            accs = predict(test, ensemble_file, 0, valid)

            result_folder = os.path.join(RESULTS_DIR, dataset_number)
            os.makedirs(result_folder, exist_ok=True)

            for file in os.listdir(DATA_DIR):
                if file.endswith(".result") and file.startswith(filename):
                    shutil.move(os.path.join(DATA_DIR, file), os.path.join(result_folder, file))
                elif file.endswith(".json") or file.endswith(".dot"):
                    os.remove(os.path.join(DATA_DIR, file))

            ensemble_result_path = "ensemble.result"
            if os.path.exists(ensemble_result_path):
                new_name = f"{dataset_number}_ensemble.result"
                shutil.move(ensemble_result_path, os.path.join(result_folder, new_name))
            print(dataset_number)

def run_boosting():
    for filename in sorted(os.listdir(DATA_DIR), key=extract_index):
        if filename.endswith("_train.txt.dat"):
            full_path = os.path.join(DATA_DIR, filename)
            base_name = full_path.replace("_train.txt.dat", "")
            dataset_number = base_name.split("\\")[-1].split("_")[0]

            train = base_name + "_train.txt.dat"
            valid = base_name + "_valid.txt.dat"
            test = base_name + "_test.txt.dat"
            ensemble_file = train + ".ff.final_boosting.json"
            apta_file = train + ".ff.final.json"
            print("Base name: ", base_name)
            print("Train: ", train)
            print("Test: ", test)
            print("Ensemble file: ", ensemble_file)
            print("Apta file: ", apta_file)

            ensemble_boosting(train, 10, valid)
            acc = predict(test, ensemble_file, 1, valid)

            result_folder = os.path.join(RESULTS_DIR, dataset_number)
            os.makedirs(result_folder, exist_ok=True)

            for file in os.listdir(DATA_DIR):
                if file.endswith(".result") and file.startswith(valid.split("\\")[-1]):
                    shutil.move(os.path.join(DATA_DIR, file), os.path.join(result_folder, file))
                elif file.endswith(".json") or file.endswith(".dot"):
                    os.remove(os.path.join(DATA_DIR, file))

            ensemble_result_path = "ensemble.result"
            if os.path.exists(ensemble_result_path):
                new_name = f"{dataset_number}_ensemble.result"
                shutil.move(ensemble_result_path, os.path.join(result_folder, new_name))
            print(dataset_number)
#run_boosting()

def plot_metric(metric_index):
    # metric_index:
    # 0 - accuracy
    # 1 - precision
    # 2 - recall
    # 3 - specificity
    # 4 - f1
    # 5 - balanced_accuracy

    densities = []
    ensemble_metrics = []
    dfa_metrics = []
    metric_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-score", "Balanced Accuracy", "Error Rate", "MCC"]
    ensemble_metrics_list = [[] for _ in ENSEMBLE_DIRS]

    dataset_folders = sorted(os.listdir(os.path.join(RESULTS_DIR, ENSEMBLE_DIRS[0])), key=lambda x: int(x))

    for dataset_folder in dataset_folders:
        dataset_number = dataset_folder
        train_file = get_train_file_path(dataset_number)
        density = compute_train_density(train_file)
        densities.append(density)

        # Get single DFA metric (assume it's the same across ensembles, use first dir)
        dfa_result_file = os.path.join(
            RESULTS_DIR, ENSEMBLE_DIRS[0], dataset_folder,
            f"{dataset_number}_training.txt_train.txt.dat.ff.final.json.result"
        )
        dfa_result = parse_results(dfa_result_file)
        dfa_metrics.append(dfa_result[metric_index])

        # Get metrics for all ensembles
        for i, ensemble_dir in enumerate(ENSEMBLE_DIRS):
            ensemble_result_file = os.path.join(
                RESULTS_DIR, ensemble_dir, dataset_folder,
                f"{dataset_number}_ensemble.result"
            )
            ensemble_result = parse_results(ensemble_result_file)
            ensemble_metrics_list[i].append(ensemble_result[metric_index])

    sorted_data = sorted(
        zip(densities, dfa_metrics, *ensemble_metrics_list),
        key=lambda x: x[0]
    )

    # Sample 50 evenly spaced points
    step = max(1, len(sorted_data) // 50)
    sampled_data = sorted_data[::step][:50]  # no more than 50 points for plot readability

    sorted_lists = list(zip(*sampled_data))
    sparsities_sorted = sorted_lists[0]
    dfa_metrics_sorted = sorted_lists[1]
    ensemble_metrics_sorted_lists = sorted_lists[2:]

    plt.figure(figsize=(14, 6))
    fts = 20
    plt.plot(sparsities_sorted, dfa_metrics_sorted, label="Single DFA", marker='o')

    ensemble_name = ["Randomized, r = 0.1", "Randomized, r = 0.4", "Randomized, r = 0.7", "Boosted"]
    for i, ensemble_metrics_sorted in enumerate(ensemble_metrics_sorted_lists):
        plt.plot(sparsities_sorted, ensemble_metrics_sorted, label=ensemble_name[i], marker='x')

    plt.xlabel("Density", fontsize=fts)
    plt.ylabel(metric_names[metric_index], fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.legend(fontsize=fts)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#plot_metric(4)
# ------------------------------------------

def load_predictions(file_path):
    with open(file_path, "r") as f:
        if ".json" in file_path:
            pred = [line.strip().split(";")[9].strip() for line in f if ";" in line]
        else:
            pred = [line.strip().split(";")[3].strip() for line in f if ";" in line]
        return pred[1:]

def compute_pairwise_disagreement(pred_lists):
    n = len(pred_lists)
    if n < 2: return 0.0
    disagreements = [
        sum(p1 != p2 for p1, p2 in zip(pred_lists[i], pred_lists[j])) / len(pred_lists[0])
        for i, j in combinations(range(n), 2)
    ]
    return sum(disagreements) / len(disagreements)

def average_disagreement_rate():
    densities = []
    ensemble_disagreements = [[] for _ in ENSEMBLE_DIRS]

    for dataset_id in range(1, 101):
        dataset_number = str(dataset_id)
        density = compute_train_density(get_train_file_path(dataset_number))
        densities.append(density)

        for i, ensemble in enumerate(ENSEMBLE_DIRS):
            dataset_path = os.path.join(RESULTS_DIR, ensemble, dataset_number)
            preds = []
            if os.path.exists(dataset_path):
                for fname in os.listdir(dataset_path):
                    if fname.endswith(".result") and (".json" in fname or ".model" in fname):
                        preds.append(load_predictions(os.path.join(dataset_path, fname)))
            disagreement = compute_pairwise_disagreement(preds) if preds else 0.0
            ensemble_disagreements[i].append(disagreement)

    # Sort datasets by sparsity
    combined = sorted(zip(densities, *ensemble_disagreements), key=lambda x: x[0])
    step = max(1, len(combined) // 50)
    sampled = combined[::step][:50]

    spars_sorted = [x[0] for x in sampled]
    disagreement_lines = [ [x[i + 1] for x in sampled] for i in range(len(ENSEMBLE_DIRS)) ]

    plt.figure(figsize=(14, 6))
    fts = 20
    labels = ["Randomized, r = 0.1", "Randomized, r = 0.4", "Randomized, r = 0.7", "Boosted"]
    #markers = ["o", "x", "^"]

    for i, disagreements in enumerate(disagreement_lines):
        plt.plot(spars_sorted, disagreements, label=labels[i])

    plt.xlabel("Density", fontsize=fts)
    plt.ylabel("Average Disagreement Rate", fontsize=fts)
    plt.xticks(fontsize=fts)
    plt.yticks(fontsize=fts)
    plt.legend(fontsize=fts)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#average_disagreement_rate()

# filename = SOURCE_FILE

# for dataset_id in range(2, 101):
#     dataset_str = str(dataset_id)
#     for ensemble in TARGET_ENSEMBLES:
#         target_dir = os.path.join(RESULTS_DIR, ensemble, dataset_str)

#         source_file = os.path.join(filename, dataset_str, f"{dataset_str}_training.txt_train.txt.dat.ff.final.json.result")
#         target_file = os.path.join(target_dir, f"{dataset_str}_training.txt_train.txt.dat.ff.final.json.result")
#         #print(source_file)
#         shutil.copy2(source_file, target_file)
#         print(f"Copied to: {target_file}")

def summarize_and_test_by_density_split(metric_index):
    densities = []
    dfa_metrics = []
    ensemble_metrics_list = [[] for _ in ENSEMBLE_DIRS]

    dataset_folders = sorted(os.listdir(os.path.join(RESULTS_DIR, ENSEMBLE_DIRS[0])), key=lambda x: int(x))

    for dataset_folder in dataset_folders:
        dataset_number = dataset_folder
        train_file = get_train_file_path(dataset_number)
        density = compute_train_density(train_file)
        densities.append(density)

        dfa_result_file = os.path.join(
            RESULTS_DIR, ENSEMBLE_DIRS[0], dataset_folder,
            f"{dataset_number}_training.txt_train.txt.dat.ff.final.json.result"
        )
        dfa_result = parse_results(dfa_result_file)
        dfa_metrics.append(dfa_result[metric_index])

        for i, ensemble_dir in enumerate(ENSEMBLE_DIRS):
            ensemble_result_file = os.path.join(
                RESULTS_DIR, ensemble_dir, dataset_folder,
                f"{dataset_number}_ensemble.result"
            )
            ensemble_result = parse_results(ensemble_result_file)
            ensemble_metrics_list[i].append(ensemble_result[metric_index])

    # Sort by sparsity
    combined = sorted(zip(densities, dfa_metrics, *ensemble_metrics_list), key=lambda x: x[0])

    midpoint = len(combined) // 2
    bottom_half = combined[:midpoint]
    upper_half = combined[midpoint:]

    def summarize_group(group, label):
        print(f"\n--- {label} datasets (n={len(group)}) ---")
        group_arrays = list(zip(*group))
        spars, dfa_arr, *ens_arrs = group_arrays

        # Report baseline
        mean_dfa = np.mean(dfa_arr)
        std_dfa = np.std(dfa_arr)
        print(f"Single DFA: mean={mean_dfa:.4f}, std={std_dfa:.4f}")

        for i, ensemble_arr in enumerate(ens_arrs):
            mean_val = np.mean(ensemble_arr)
            std_val = np.std(ensemble_arr)
            print(f"{ENSEMBLE_DIRS[i]}: mean={mean_val:.4f}, std={std_val:.4f}")

            t_stat, p_val = ttest_rel(ensemble_arr, dfa_arr)
            print(f"  vs DFA: t={t_stat:.2f}, p={p_val:.4f}")

    summarize_group(bottom_half, "Low density (bottom half)")
    summarize_group(upper_half, "High density (upper half)")
    summarize_group(combined, "All datasets")

#summarize_and_test_by_density_split(4)