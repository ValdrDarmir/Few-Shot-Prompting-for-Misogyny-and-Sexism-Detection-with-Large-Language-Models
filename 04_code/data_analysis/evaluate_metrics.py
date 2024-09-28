# This script analysis the models with typical metrics accuracy, recall,
# precision and micro-f1
# Calculation from:
# Simmering, P. F., & Huoviala, P. (2023).
# Large language models for aspect-based sentiment analysis (arXiv:2310.18025)
# . arXiv. https://doi.org/10.48550/arXiv.2310.18025

# Author: Niklas Donhauser
# Date: September 05, 2024

# import libraries
import sys
import json
import numpy as np


# calculate precision
def precision(tp, fp):
    precision = np.where((tp + fp) == 0, 0, tp / (tp + fp))
    print(f"Precision: {precision.astype(float)}")
    return precision.astype(float)


# calculate recall
def recall(tp, fn):
    recall = np.where((tp + fn) == 0, 0, tp / (tp + fn))
    print(f"Recall: {recall.astype(float)}")
    return recall.astype(float)


# calculate f1 score
def f1_score(tp, fp, fn):
    pre = precision(tp, fp)
    rec = recall(tp, fn)
    f1 = np.where((pre + rec) == 0, 0, 2 * (pre * rec) / (pre + rec))
    print(f"F1: {f1.astype(float)}")
    return f1.astype(float)


# calculate accuracy
def accuracy(tp, fp, fn):
    acc = np.where((tp + fp + fn) == 0, 0, tp / (tp + fp + fn))
    return acc.astype(float)


# compare the gold labels with the prediction labels
def compareValues(corpus_dict, prediction_dict, output_file):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for key_corpus in corpus_dict:
        for key_prediction in prediction_dict:
            if key_corpus == key_prediction:
                tp, fp, fn = comparePrediction(
                    corpus_dict[key_corpus], prediction_dict[key_corpus])

                tp_total = tp_total + tp
                fp_total = fp_total + fp
                fn_total = fn_total + fn
                # print(corpus_dict[key_corpus])
    print(tp_total, fp_total, fn_total)
    saveData(tp_total, fp_total, fn_total, prediction_dict, output_file)


# compare one entry; saved in a set to compare, return the fp, tp and fn
def comparePrediction(corpus_items, prediction_items):
    corpus_set = set()
    prediction_set = set()
    for item_corpus in corpus_items:
        user = item_corpus['user']
        label = item_corpus['label']
        corpus_set.update([(user, label)])
    for item_prediction in prediction_items:
        user = item_prediction['user']
        label = item_prediction['label']
        prediction_set.update([(user, label)])
    # print(corpus_set)
    # print(prediction_set)
    tp = len(corpus_set.intersection(prediction_set))
    fp = len(prediction_set.difference(corpus_set))
    fn = len(corpus_set.difference(prediction_set))

    if len(corpus_set) == 0 and len(prediction_set) == 0:
        tp += 1

    # print(tp, fp, fn)
    return tp, fp, fn


# transform the jsonl file into a dict with the id as key and all annotations
# with name and label as value
def make_dict(file_path):
    result_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line.strip())

            id_ = json_obj['id']
            annotations = json_obj['annotations']

            result_dict[id_] = annotations
    return result_dict


# save the data in a file
def saveData(tp, fp, fn, prediction_dict, output_file):
    rec = recall(tp, fn)
    pre = precision(tp, fp)
    f1 = f1_score(tp, fp, fn)
    acc = accuracy(tp, fp, fn)
    metrics = f"Precision: {pre} \nRecall: {
        rec} \nF1-Score: {f1} \nAccuracy: {acc}\n"
    data = f"False Negative: {fn} \nTrue Positive {tp} \nFalse Positive {
        fp} \nAmount Predictions(): {len(prediction_dict)}\n"  # noqa: E501
    message = metrics + data
    with open(f"../../05_results/{output_file}", "w") as file:
        file.write(message)


def main():
    print("Start")
    prediction = "../../03_input/[method]/result.jsonl"  # noqa: E501
    output_file = "[methodname].txt"
    gold = "data/competition/germeval-competition-merged.jsonl"
    prediction_dict = make_dict(prediction)
    gold_dict = make_dict(gold)
    print(len(prediction_dict))
    compareValues(gold_dict, prediction_dict, output_file)


if __name__ == '__main__':
    sys.exit(main())
