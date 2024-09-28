# This script combines the target json file with the test.jsonl file to store
# the information in the same manner as the trainingsfile.
# relevant for trial phase
# similar as merge_labels_for_testset.py

# Author: Niklas Donhauser
# Date: August 22, 2024

# import libraries
import json
import jsonlines
import sys


# combine files for a simpler evaluation. testset with addional gold labels
def combine_files(targets_file, test_file, output_jsonl_file_path):
    with open(targets_file, 'r') as file:
        json_data = json.load(file)

    # create mapping
    id_to_labels = {entry['id']: entry['labels'] for entry in json_data}

    # add labels
    with jsonlines.open(test_file, mode='r') as reader, jsonlines.open(output_jsonl_file_path, mode='w') as writer:  # noqa: E501
        for line in reader:
            entry_id = line['id']
            if entry_id in id_to_labels:
                labels = id_to_labels[entry_id]
                annotations = [{"user": user, "label": label}
                               for user, label in zip(line['annotators'],
                                                      labels)]
                updated_line = {
                    "id": entry_id,
                    "text": line['text'],
                    "annotations": annotations
                }
                writer.write(updated_line)


def main():
    targets_file = "[path]/germeval-trial-targets.json"
    test_file = "[path]/germeval-trial-test.jsonl"
    output_jsonl_file_path = "[path]/merged.jsonl"
    combine_files(targets_file, test_file, output_jsonl_file_path)


if __name__ == '__main__':
    sys.exit(main())
