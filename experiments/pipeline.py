import subprocess
import re
import csv
from math import sqrt

flexfringe_path = "./cmake-build-debug-visual-studio/flexfringe"

def ensemble_random(input_file, nr_estimators, lb, ub):
    input = input_file
    options_ensemble = ["--ini=ini/edsm.ini", "--mode", "ensemblerandom", f"--estimators={nr_estimators}", f"--ensemblingrandomlb={lb}", f"--ensemblingrandomub={ub}", input]
    cmd = [flexfringe_path] + options_ensemble
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("=== STDOUT ===")
        print(result.stdout)

        print("\n=== STDERR ===")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print("FlexFringe execution failed:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)

def ensemble_boosting(input_file, nr_estimators, valid_file):
    input = input_file
    options_ensemble = ["--ini=ini/edsm.ini", "--mode", "ensembleboosting", f"--estimators={nr_estimators}", "--boosting=1", f"--validfile={valid_file}", input]
    cmd = [flexfringe_path] + options_ensemble
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("=== STDOUT ===")
        print(result.stdout)

        print("\n=== STDERR ===")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print("FlexFringe execution failed:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)

def predict(input_file, ensemble_file, boosting, valid):
    test = input_file
    options_predict = ["--ini=ini/edsm.ini", "--mode", "predictens", "--predicttype", "1", f"--boosting={boosting}", test, f"--aptafile={ensemble_file}", f"--validfile={valid}"]
    cmd = [flexfringe_path] + options_predict
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("=== STDOUT ===")
        print(result.stdout)

        print("\n=== STDERR ===")
        print(result.stderr)

        model_accs = re.findall(r"model \d+ acc ([0-9.]+)", result.stdout)
        ensemble_acc = re.search(r"Ensemble accuracy: ([0-9.]+)", result.stdout)
        accuracies = [float(a) for a in model_accs]
        if ensemble_acc:
            accuracies.append(float(ensemble_acc.group(1)))

        print(accuracies)
        return accuracies

    except subprocess.CalledProcessError as e:
        print("FlexFringe execution failed:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)

def learn_single_dfa(input_file):
    options_learn = ["--ini=ini/edsm.ini",  input_file]
    cmd = [flexfringe_path] + options_learn
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        print("=== STDOUT ===")
        print(result.stdout)

        print("\n=== STDERR ===")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print("FlexFringe execution failed:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)

def predict_single_dfa(input_file, apta_file):
    test = input_file
    options_predict = ["--ini=ini/edsm.ini", "--mode", "predict", "--predicttype", "1", test, f"--aptafile={apta_file}"]
    cmd = [flexfringe_path] + options_predict
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        res_file = apta_file + ".result"
        correct = 0
        total = 0

        with open(res_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                true_type = row[" trace type"].strip()
                pred_type = row[" predicted trace type"].strip()
                
                if true_type == pred_type:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy}")
        return accuracy

    except subprocess.CalledProcessError as e:
        print("FlexFringe execution failed:")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


def parse_results(res_file):
    TP = TN = FP = FN = 0
    total = 0

    with open(res_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            true_label = int(row[" trace type"].strip())
            predicted = int(row[" predicted trace type"].strip())
            total += 1
            if true_label == 1 and predicted == 1:
                TP += 1
            elif true_label == 0 and predicted == 0:
                TN += 1
            elif true_label == 0 and predicted == 1:
                FP += 1
            elif true_label == 1 and predicted == 0:
                FN += 1

    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    return (accuracy, precision, recall, specificity, f1, balanced_accuracy)