# This script transforms the predictions into the format needed for the
# competition subtask 2(.tsv file)

# Author: Niklas Donhauser
# Date: September 20, 2024

# import libraries
import json
import pandas as pd


# Function to calculate the distributions
def calculate_distributions(annotations):
    total_annotations = len(annotations)

    counts = {
        '0-Kein': 0,
        '1-Gering': 0,
        '2-Vorhanden': 0,
        '3-Stark': 0,
        '4-Extrem': 0
    }

    # Count labels
    for annotation in annotations:
        label = annotation['label']
        counts[label] += 1

    # Calculate the distribution multi
    dist_multi = [
        counts['0-Kein'] / total_annotations,
        counts['1-Gering'] / total_annotations,
        counts['2-Vorhanden'] / total_annotations,
        counts['3-Stark'] / total_annotations,
        counts['4-Extrem'] / total_annotations
    ]

    # Calculate the distribution binary
    dist_bin = [
        counts['0-Kein'] / total_annotations,
        (counts['1-Gering'] + counts['2-Vorhanden'] +
         counts['3-Stark'] + counts['4-Extrem']) / total_annotations
    ]

    return dist_bin, dist_multi


# Load the file and calculate the needed information for the scoring.
def process_jsonl_file(input_file):
    data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            id_ = record['id']
            annotations = record['annotations']

            dist_bin, dist_multi = calculate_distributions(annotations)

            # append the data
            data.append({
                'id': id_,
                'dist_bin_0': dist_bin[0],
                'dist_bin_1': dist_bin[1],
                'dist_multi_0': dist_multi[0],
                'dist_multi_1': dist_multi[1],
                'dist_multi_2': dist_multi[2],
                'dist_multi_3': dist_multi[3],
                'dist_multi_4': dist_multi[4]
            })

    return data


# Convert the processed data to a TSV file
def save_to_tsv(data, output_file):
    df = pd.DataFrame(data)
    df.to_csv(output_file, sep='\t', index=False)


def main():
    input_file = "../../03_input/[model]/result.jsonl"
    output_file = "../../03_input/[model]/results_st2.tsv"
    data = process_jsonl_file(input_file)
    save_to_tsv(data, output_file)
    print(f"Transformation (ST2) complete for file {input_file}")


if __name__ == '__main__':
    main()
