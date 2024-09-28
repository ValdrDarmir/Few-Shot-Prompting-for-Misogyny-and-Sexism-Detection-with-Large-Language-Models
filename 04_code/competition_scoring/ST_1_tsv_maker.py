# This script transforms the predictions into the format needed for the
# competition subtask 1 (.tsv file)

# Author: Niklas Donhauser
# Date: September 20, 2024

# import libraries
import json
import csv
from collections import Counter

# mapping for the labels into numeric values
label_mapping = {
    "0-Kein": 0,
    "1-Gering": 1,
    "2-Vorhanden": 2,
    "3-Stark": 3,
    "4-Extrem": 4
}


# calculates the metrics for the competition
def calculate_metrics(entry):
    annotations = entry['annotations']
    labels = [label_mapping[annotation['label']] for annotation in annotations]
    label_counts = Counter(labels)

    # Determine the majority label
    majority_label = label_counts.most_common(1)[0][0]

    # Calculate other metrics
    bin_maj = 1 if majority_label != 0 else 0
    bin_one = 1 if any(label != 0 for label in labels) else 0
    bin_all = 1 if all(label != 0 for label in labels) else 0

    multi_maj = [k for k, v in label_mapping.items() if v == majority_label][0]

    disagree_bin = 1 if any(label != 0 for label in labels) and any(
        label == 0 for label in labels) else 0

    return {
        "id": entry["id"],
        "bin_maj": bin_maj,
        "bin_one": bin_one,
        "bin_all": bin_all,
        "multi_maj": multi_maj,
        "disagree_bin": disagree_bin
    }


# opens the file and process it
def process_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            metrics = calculate_metrics(entry)
            results.append(metrics)

    return results


# save the results
def save_results(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, delimiter='\t', fieldnames=[
            "id", "bin_maj", "bin_one", "bin_all", "multi_maj", "disagree_bin"
        ])
        writer.writeheader()
        writer.writerows(results)


# main function, change input_file and outfile to the right folder
def main():
    input_file = "../../03_input/[model]/result.jsonl"
    output_file = "../../03_input/[model]/results_st1.tsv"
    results = process_file(input_file)
    save_results(results, output_file)
    print(f"Transformation (ST1) complete for file {input_file}")


if __name__ == '__main__':
    main()
