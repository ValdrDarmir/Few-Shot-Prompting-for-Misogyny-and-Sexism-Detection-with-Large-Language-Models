# This script generates wordclouds

# Author: Niklas Donhauser
# Date: September 05, 2024

# import libraries
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import os
import sys
# import nltk
# nltk.download('stopwords')


# generate the wordcloud for every label a new wordcloud
def generate_wordcloud(file_name, setname):
    file_path = os.path.join('../../01_data', file_name)
    text_data_by_label = defaultdict(set)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            text = entry['text']
            labels = [annotation['label']
                      for annotation in entry['annotations']]
            for label in labels:
                text_data_by_label[label].add(text)

    text_data_by_label = {label: " ".join(
        texts) for label, texts in text_data_by_label.items()}

    print(text_data_by_label["0-Kein"])
    german_stopwords = set(stopwords.words('german'))
    for label, text_data in text_data_by_label.items():
        # Tokenize the text and remove stopwords
        words = [
            word.lower() for word in text_data.split()
            if word.lower() not in german_stopwords and word.isalpha()
        ]

        # Count word frequencies
        word_counter = Counter(words)

        most_common_words = word_counter.most_common(80)

        print(f"Top 10 words for label '{label}':")
        for word, count in most_common_words:
            print(f"{word}: {count}")
        print("\n" + "-"*40 + "\n")
    # Generate the word cloud for each label
    for label, text_data in text_data_by_label.items():
        wordcloud = WordCloud(width=800, height=400, max_words=200,
                              background_color='white',
                              stopwords=german_stopwords).generate(text_data)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        plt.figtext(0.5, 0.01, f"Most common words for the label: {label} ({setname})",  # noqa: E501
                    ha="center", fontsize=24)

        print(file)
        short_file_name = file_name[:-6]
        name = short_file_name + "_" + label

        output_path_svg = f'../../05_results/visuals/{name}.svg'
        plt.savefig(output_path_svg, format='svg')

        output_path_png = f'../../05_results/visuals/{name}.png'
        plt.savefig(output_path_png, format='png')
        plt.close()


def main():
    # exchange setname
    files = ["competition_test.jsonl"]
    setname = "rainset"
    for file in files:
        generate_wordcloud(file, setname)


if __name__ == '__main__':
    sys.exit(main())
