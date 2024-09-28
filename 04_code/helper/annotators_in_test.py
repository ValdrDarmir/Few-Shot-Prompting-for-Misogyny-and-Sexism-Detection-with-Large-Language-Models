# This script counts the unique annotators per file
# Author: Niklas Donhauser
# Date: September 10, 2024

# import libraries
import json
import sys
from collections import defaultdict


# counts the unique annotators in a file
def count_annotator_appearances(file_path):
    annotator_counts = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            annotators = entry.get('annotators', [])
            for annotator in annotators:
                annotator_counts[annotator] += 1

    return annotator_counts


def main():
    # update path for the file
    file_name = '../../01_data/[name_of_the_set].jsonl'
    annotator_count = count_annotator_appearances(file_name)
    print(f'Total unique annotators: {annotator_count}')


if __name__ == '__main__':
    sys.exit(main())
