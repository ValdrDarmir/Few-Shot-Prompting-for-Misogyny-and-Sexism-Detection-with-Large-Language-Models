# This script "generates" examples for the few shot method. Extracting examples
# out of the trainset

# Author: Niklas Donhauser
# Date: September 21, 2024

# import libraries
import json
import os
import sys
import random
from collections import defaultdict


# load the data
def load_jsonl(file_path, encoding='utf-8'):
    data = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data


# save the examples
def save_jsonl(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')


# get random examples by label
def select_n_examples_per_label(data, n, max_chars=150):
    label_buckets = defaultdict(list)

    # Group data by label and filter by text length
    for entry in data:
        label = entry['annotations'][0]['label']
        if len(entry['text']) <= max_chars:
            label_buckets[label].append(entry)

    selected_examples = []
    # Randomly select n examples from each label bucket
    for label, examples in label_buckets.items():
        if len(examples) < n:
            print("Warning! Not enough examples")
            selected_examples.extend(examples)
        else:
            selected_examples.extend(random.sample(examples, n))

    return selected_examples


# setup for the extraction; needed jsonl with splitted annotators
# see splitting_traindata.py
def main():
    examples_amount = 2  # change for the amount of examples: possible values
    # 1,2
    base_path = '[path]/[annotators_dir]'
    output_base_path = f'[path]/{examples_amount*5}_examples'
    # A001 - A010, excluding A006
    annotators = [f'A00{i}' if i <
                  10 else 'A010' for i in range(1, 11) if i != 6]

    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    for annotator in annotators:
        input_file_path = os.path.join(base_path, f'{annotator}.jsonl')
        output_file_path = os.path.join(output_base_path, f'{annotator}.jsonl')

        data = load_jsonl(input_file_path)

        if not data:
            print(f"No data found for {annotator}.")
            continue

        selected_examples = select_n_examples_per_label(data, examples_amount)

        # Save the selected examples
        save_jsonl(output_file_path, selected_examples)
        print(f'Saved {len(selected_examples)} examples to {output_file_path}')


if __name__ == "__main__":
    sys.exit(main())
