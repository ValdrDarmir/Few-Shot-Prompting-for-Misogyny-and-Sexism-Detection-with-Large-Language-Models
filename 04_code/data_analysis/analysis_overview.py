# This script analysis the different .jsonl files and displays the entries,
# avg. words per entry and max and min entry length

# Author: Niklas Donhauser
# Date: August 22, 2024

# import libraries
import os
import sys
import json


# generate the values per file
def generate_overview(file_name):
    file_path = os.path.join('corpus', file_name)

    total_text_length = []
    min_text = None
    max_text = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            text = data['text']
            text_length = len(text)

            # display the longest and shortest entry
            if min_text is None or text_length < len(min_text):
                min_text = text
            if max_text is None or text_length > len(max_text):
                max_text = text

            total_text_length.append(text_length)

    average_length = sum(total_text_length) / \
        len(total_text_length) if total_text_length else 0

    print(file_name)
    print(f"Texts: {len(total_text_length)}")
    print(f"Average length of text: {average_length:.2f} characters")
    print(f"Minimum length text: {
        len(min_text)} characters -> Text: {min_text}")
    print(f"Maximum length text: {len(max_text)} characters")
    print("\n" + "-"*40 + "\n")


def main():
    # change filenames for the different sets (train/test
    # and competition/trial)
    # put data in 01_data
    files = ["[filename].jsonl"]
    for file in files:
        generate_overview(file)


if __name__ == '__main__':
    sys.exit(main())
