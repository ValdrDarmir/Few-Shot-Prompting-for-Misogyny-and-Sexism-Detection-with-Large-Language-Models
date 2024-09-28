# This script splits the trainset into annotatorsets where each set only
# contains annotations by the same annotator
# (needed for examples (generate_examples.py))

# Author: Niklas Donhauser
# Date: September 05, 2024

# import libraries
import json
import os
import sys

# mapping labels into numeric values
label_order = {
        "0-Kein": 0,
        "1-Gering": 1,
        "2-Vorhanden": 2,
        "3-Stark": 3,
        "4-Extrem": 4
    }


# split the data by annotator
def split_data(input_file, output_dir):
    annotator_data = {f"A{str(i).zfill(3)}": [] for i in range(1, 11)}

    # Read the input file and split data by annotator
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry['text']
            annotations = entry['annotations']
            id_ = entry['id']

            # Separate annotations by annotator
            for annotation in annotations:
                user = annotation['user']
                if user in annotator_data:
                    # Append entry with only the relevant annotation
                    annotator_data[user].append({
                        'id': id_,
                        'text': text,
                        'annotations': [{'user': user, 'label':
                                        annotation['label']}]
                    })

    # Sort and write the data to separate files for each annotator
    for annotator, data in annotator_data.items():
        # Sort the entries by label
        data.sort(key=lambda x: label_order[x['annotations'][0]['label']])

        output_file = os.path.join(output_dir, f'{annotator}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("Data split by annotator, sorted by label, and saved successfully.")


def main():
    input_file = '[path]/[file_name].jsonl'
    output_dir = '[path]'
    split_data(input_file, output_dir)


if __name__ == '__main__':
    sys.exit(main())
