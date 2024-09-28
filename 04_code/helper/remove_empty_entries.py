# This script removes empty entries if the API response contains empty
# annotations

# Author: Niklas Donhauser
# Date: September 05, 2024

# import libraries
import sys
import json


# remove empty annotations entries
def remove_empty_annotations(input_file, output_file):
    with open(input_file,
              'r',
              encoding='utf-8') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            if 'annotations' in entry and entry['annotations'] != []:
                json.dump(entry, outfile)
                outfile.write('\n')


def main():
    input_jsonl_file = '[path]/result.jsonl'
    output_jsonl_file = '[path]/output_name.jsonl'
    remove_empty_annotations(input_jsonl_file, output_jsonl_file)


if __name__ == '__main__':
    sys.exit(main())
