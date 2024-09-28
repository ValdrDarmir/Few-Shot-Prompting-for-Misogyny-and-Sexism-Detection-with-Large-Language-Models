# This script combines the targets (json) with the test set (jsonl) for the
# evaluation and brings it to the same format as the trial and development
# phase
# Author: Niklas Donhauser
# Date: September 28, 2024

# import libraries
import json
import jsonlines
import sys


# merge the two files and add the labels to the jsonl file
def merge_files(json_file_path, jsonl_file_path, output_jsonl_file_path):
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)

    # Create a mapping from id to labels
    id_to_labels = {entry['id']: entry['labels'] for entry in json_data}

    # Process the jsonl file and enrich it with labels
    with jsonlines.open(jsonl_file_path, mode='r') as reader, jsonlines.open(output_jsonl_file_path, mode='w') as writer:  # noqa: E501
        for obj in reader:
            entry_id = obj['id']
            if entry_id in id_to_labels:
                labels = id_to_labels[entry_id]
                annotations = [{"user": user, "label": label}
                               for user, label in zip(obj['annotators'],
                                                      labels)]
                enriched_obj = {
                    "id": entry_id,
                    "text": obj['text'],
                    "annotations": annotations
                }
                writer.write(enriched_obj)


def main():
    json_file_path = "[path]/germeval-competition-targets.json"
    jsonl_file_path = "[path]/germeval-competition-test_org.jsonl"
    output_jsonl_file_path = "[path]/germeval-competition-merged.jsonl"
    merge_files(json_file_path, jsonl_file_path, output_jsonl_file_path)


if __name__ == "__main__":
    sys.exit(main())
