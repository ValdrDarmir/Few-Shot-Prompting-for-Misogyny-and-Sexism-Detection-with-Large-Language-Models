# This script analysis the different .jsonl files and counts the different
# labels and their occurrences per file.

# Author: Niklas Donhauser
# Date: August 22, 2024

# import libraries
import json
import sys
import os
from collections import Counter

# order of the labels
label_order = ["0-Kein", "1-Gering", "2-Vorhanden", "3-Stark", "4-Extrem"]


# counts the labels per file and prints them
def count_label_appearances(file_name):
    file_path = os.path.join('corpus', file_name)
    label_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            # counts the labels in the annotations element
            for annotation in data.get('annotations', []):
                label = annotation.get('label')
                if label:
                    label_counts[label] += 1

    print(file_name)
    for label in label_order:
        print(f"{label}: {label_counts[label]}")

    total = 0
    for label, count in label_counts.items():
        total += int(count)

    print(f"Total: {total}")
    print("\n" + "-"*40 + "\n")


def main():
    # change filenames for the different sets (train/test
    # and competition/trial)
    # put data in 01_data
    files = ["[filename].jsonl"]
    for file in files:
        count_label_appearances(file)


if __name__ == '__main__':
    sys.exit(main())
