# This script makes a new testset from the trainset for model performance check

# Author: Niklas Donhauser
# Date: August 15, 2024

# import libraries
import json
import sys


# splits the data
def makeSplit(input_file, output_file):
    split_ratio = 0.05
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    test_size = int(len(data) * split_ratio)

    # Split the data
    test_data = data[:test_size]

    with open(output_file, 'w', encoding='utf-8') as file:
        for entry in test_data:
            file.write(json.dumps(entry) + '\n')


def main():
    input_file = '[path]/germeval-competition-traindev.jsonl'
    output_file = '[path]/test_split.jsonl'
    makeSplit(input_file, output_file)


if __name__ == '__main__':
    sys.exit(main())
