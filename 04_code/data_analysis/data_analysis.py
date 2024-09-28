# This script analysis the corpus in many different ways

# Author: Niklas Donhauser
# Date: September 05, 2024

# import libraries
import json
import nltk
import sys
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
nltk.download('stopwords')

# dictionaries and lists to store data
text_lengths = []
annotator_counts = []
annotator_scores = {}
annotation_counts = {}

# transform labels into numeric values
label_mapping = {
    "0-Kein": 0,
    "1-Gering": 1,
    "2-Vorhanden": 2,
    "3-Stark": 3,
    "4-Extrem": 4
}


# basic analysis for the jsonl file: Text length (max, min, average)
def basic_analysis(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            text = entry['text']
            text_length = len(text)
            text_lengths.append(text_length)

    max_length = max(text_lengths)
    min_length = min(text_lengths)
    average_length = sum(text_lengths) / len(text_lengths)
    median_length = sorted(text_lengths)[len(text_lengths) // 2]

    print(f"Text analysis for {file_path}")
    print(f"Maximum length: {max_length}")
    print(f"Minimum length: {min_length}")
    print(f"Average length: {average_length}")
    print(f"Median length: {median_length}")


# analysis the amount of annotators per text unit
def annotator_analysis(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            annotations = entry['annotations']
            annotator_count = len(annotations)
            annotator_counts.append(annotator_count)

    max_annotators = max(annotator_counts)
    min_annotators = min(annotator_counts)
    average_annotators = sum(annotator_counts) / len(annotator_counts)

    print(f"Annotator analysis for {file_path}")
    print(f"Maximum number of annotators: {max_annotators}")
    print(f"Minimum number of annotators: {min_annotators}")
    print(f"Average number of annotators: {average_annotators}")


# analysis the scores given by the different annotators per set.
# Amount of annotations, average rating, max and min rating
def annotator_score_analysis(file_path):
    # Read the JSONL file and process each line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            annotations = entry['annotations']
            for annotation in annotations:
                user = annotation['user']
                label = annotation['label']
                score = label_mapping[label]
                if user not in annotator_scores:
                    annotator_scores[user] = []
                    annotation_counts[user] = 0
                annotator_scores[user].append(score)
                annotation_counts[user] += 1

    print(f"Annotator score analysis for {file_path}")
    for user, scores in annotator_scores.items():
        max_score = max(scores)
        min_score = min(scores)
        average_score = sum(scores) / len(scores)
        num_annotations = annotation_counts[user]

        sorted_numbers = sorted(scores)
        n = len(sorted_numbers)
        if n % 2 == 1:
            median = sorted_numbers[n // 2]
        else:
            median = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2

        print(f"Annotator {user}:")
        print(f"  Number of annotations: {num_annotations}")
        print(f"  Maximum score: {max_score}")
        print(f"  Minimum score: {min_score}")
        print(f"  Average score: {average_score}")
        print(f"  Median score: {median}")


# generate a wordcloud for the different sets split by labels
def generate_wordcloud(file_path):
    text_data_by_label = defaultdict(str)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            text = entry['text']
            labels = [annotation['label']
                      for annotation in entry['annotations']]
            for label in labels:
                text_data_by_label[label] += " " + text

    german_stopwords = set(stopwords.words('german'))

    # Generate the word cloud for each label
    for label, text_data in text_data_by_label.items():
        wordcloud = WordCloud(width=800, height=400, max_words=600,
                              background_color='white',
                              stopwords=german_stopwords).generate(text_data)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        output_path_svg = f'../../05_results/visuals/{label}.svg'
        plt.savefig(output_path_svg, format='svg')

        output_path_png = f'../../05_results/visuals/{label}.png'
        plt.savefig(output_path_png, format='png')
        plt.close()


# calculates labels per annotator
def count_labels_per_annotator(file_path):
    label_counts = defaultdict(lambda: defaultdict(int))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            annotations = entry.get('annotations', [])
            for annotation in annotations:
                annotator = annotation['user']
                label = annotation['label']
                label_counts[annotator][label] += 1

    for annotator, counts in label_counts.items():
        print(f'Annotator {annotator}:')
        for label, count in counts.items():
            print(f'  {label}: {count}')


# counts the amount of labels (plus total labels)
def count_label_appearances(file_path):
    label_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            for annotation in data.get('annotations', []):
                label = annotation.get('label')
                if label:
                    label_counts[label] += 1

    for label, count in label_counts.items():
        print(f"{label}: {count}")
    total = 0
    for label, count in label_counts.items():
        total += int(count)
    print(f"Total: {total}")


# generate figures for the distribution of labels per annotator with displaying
# the mean of each annotator
def generate_annotator_distribution(file_path, name):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    annotator_data = []

    for entry in data:
        for annotation in entry['annotations']:
            annotator_data.append({
                'user': annotation['user'],
                'label': annotation['label']
            })

    df = pd.DataFrame(annotator_data)

    # Convert labels to numeric values
    df['numeric_label'] = df['label'].str.extract(r'(\d+)').astype(int)

    sorted_annotators = sorted(df['user'].unique(), key=lambda x: int(x[1:]))

    palette = sns.color_palette("Set2", n_colors=len(sorted_annotators))

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='user', y='numeric_label', data=df,
                order=sorted_annotators, palette=palette)

    means = df.groupby('user')['numeric_label'].mean()

    # Add the mean values as text inside each box
    for i, annotator in enumerate(sorted_annotators):
        mean_val = means[annotator]
        plt.text(i, mean_val, f'{mean_val:.2f}', horizontalalignment='center',
                 color='black', weight='bold')

    # Add labels and title
    plt.xlabel('Annotator')
    plt.ylabel('Label')

    output_path_png = f'../../05_results/visuals/annotator_distribution_{name}.png'  # noqa: E501
    plt.savefig(output_path_png, format='png')

    output_path_pdf = f'../../05_results/visuals/annotator_distribution_{name}.pdf'  # noqa: E501
    plt.savefig(output_path_pdf, format='pdf', dpi=300)

    plt.show()


#  makes a figure that displays the labels (amount) per annotator in a barplot
def generate_label_graph(file_path, name):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    annotator_data = []

    for entry in data:
        for annotation in entry['annotations']:
            annotator_data.append({
                'user': annotation['user'],
                'label': annotation['label']
            })

    df = pd.DataFrame(annotator_data)

    # Count the number of each label per annotator
    label_counts = df.pivot_table(
        index='user', columns='label', aggfunc='size', fill_value=0)

    # Sort annotators by the natural order (A001, A002, ..., A012)
    sorted_annotators = sorted(label_counts.index, key=lambda x: int(x[1:]))

    # Sort the label counts according to sorted annotators
    label_counts = label_counts.loc[sorted_annotators]

    ax = label_counts.plot(kind='bar', stacked=True,
                           figsize=(10, 6), colormap='Set2')

    for i in range(label_counts.shape[0]):
        cumulative_height = 0
        for j in range(label_counts.shape[1]):
            count = label_counts.iloc[i, j]
            if count > 0:
                # Get the center of the bar segment
                ax.text(i, cumulative_height + count / 2,
                        f'{count}', ha='center', va='center', color='black',
                        fontsize=9)
            cumulative_height += count

    plt.xlabel('Annotator')
    plt.ylabel('Count of Labels')

    # Save the figure as PDF and PNG
    output_path_png = f'../../05_results/visuals/annotator_distribution_chart_{name}.png'  # noqa: E501
    plt.savefig(output_path_png, format='png')

    output_path_pdf = f'../../05_results/visuals/annotator_distribution_chart_{name}.pdf'  # noqa: E501
    plt.savefig(output_path_pdf, format='pdf', dpi=300)

    plt.show()


def main():
    # change the name for the different corpus splits
    trainset = "../../01_data/[name].jsonl"
    output_name = "trainset"
    # comment/uncomment the different methods to analyze the corpus
    # testset = "../../01_data/[name].jsonl"
    # basic_analysis(testset)
    # annotator_analysis(testset)
    # annotator_score_analysis(testset)
    # generate_wordcloud(testset)
    # count_labels_per_annotator(trainset)
    # count_label_appearances(trainset)
    # generate_annotator_distribution(trainset, name)
    generate_label_graph(trainset, output_name)


if __name__ == '__main__':
    sys.exit(main())
