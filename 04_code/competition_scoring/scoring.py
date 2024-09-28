# File updated to windows paths (main)
# Author for this part Niklas Donhauser
# Date: September 20, 2024

# Author calculations (rest of the file): OFAI

# This file is licensed under the Apache 2.0 License terms, see
# https://www.apache.org/licenses/LICENSE-2.0


# see https://github.com/codalab/codabench/wiki/Competition-Bundle-Structure#scoring-program  # noqa: E501

import sys
import os
import json
import csv
from collections import defaultdict
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial import distance

GLOBALS = dict(debug=False)
EPS = 0.001
# small value to allow for rounding errors when checking for a valid
# distribution that sums to 1.0

MULT_LABELS = ["0-Kein", "1-Gering", "2-Vorhanden", "3-Stark", "4-Extrem"]

ST1_COLUMNS = ['id', 'bin_maj', 'bin_one',
               'bin_all', 'multi_maj', 'disagree_bin']
ST2_COLUMNS = ['id', 'dist_bin_0', 'dist_bin_1', 'dist_multi_0',
               'dist_multi_1', 'dist_multi_2', 'dist_multi_3', 'dist_multi_4']


def load_targets(targets_file):
    with open(targets_file, "rt", encoding="utf-8") as infp:
        targets = json.load(infp)
    return targets


def check_columns(data, columns):
    """Check if the expected columns and only the expected columns are present
    in the data, if not, print an error message to
    stderr and throw an exception. Otherwise return to the caller."""
    for column in columns:
        if column not in data:
            raise ValueError(
                f"Column {column} not found in data, got {data.keys()}")
    for column in data.keys():
        if column not in columns:
            raise ValueError(
                f"Column {column} not expected in data, expected {columns}")


def check_allowed(data, column, allowed=None):
    """Check if the predictions are in the allowed set, if not, print an error
    message to stderr also showing the id and throw an exception. Otherwise
    return to the caller."""
    if allowed is None:
        allowed = ["0", "1"]
    if column not in data:
        raise ValueError(f"Column {column} not found in data")
    for i, value in enumerate(data[column]):
        if value not in allowed:
            raise ValueError(f"Invalid value {value} not one of {allowed} in column {  # noqa: E501
                             column} at index {i} with id {data['id'][i]}")
    print(f"Column {column} is OK")


def check_dist(data, columns):
    """Check if the predictions are in the allowed range, if not, print an
    error message to stderr also showing the id and throw an exception.
    Otherwise return to the caller."""
    for column in columns:
        if column not in data:
            raise ValueError(f"Column {column} not found in data")
    for i in range(len(data["id"])):
        sum = 0.0
        theid = data["id"][i]
        for column in columns:
            try:
                value = float(data[column][i])
            except ValueError:
                raise ValueError(f"Invalid value {data[column][i]} not a float in column {  # noqa: E501
                                 column} at index {i} with id {theid}")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Invalid value {value} not in range [0.0, 1.0] in column {  # noqa: E501
                                 column} at index {i} with id {data['id'][i]}")
            sum += value
        if abs(sum - 1.0) > EPS:
            raise ValueError(f"Values in columns {
                             columns} do not sum to 1.0 at index {
                                 i} with id {theid}")


def load_tsv(submission_dir, expected_rows, expected_cols, file=None):
    """
    Try to load a TSV file from the submission directory. This expects a
    single TSV file to be present in the submission directory.
    If there is no TSV file or there are multiple files, it will log an error
    to stderr and return None.
    """
    if file is not None:
        tsv_file = file
    else:
        tsv_files = [f for f in os.listdir(
            submission_dir) if f.endswith('.tsv')]
        if len(tsv_files) == 0:
            print(
                "No TSV file ending with '.tsv' found in submission directory",
                file=sys.stderr)
            return None
        if len(tsv_files) > 1:
            print("Multiple TSV files found in submission directory",
                  file=sys.stderr)
            return None
        tsv_file = tsv_files[0]
    tsv_path = os.path.join(submission_dir, tsv_file)
    print("Loading TSV file", tsv_path)
    # Read the TSV file incrementally row by row and create a dictionary where
    # the key is the column name and the value is a list of values for that
    # column.
    # Expect the column names in the first row of the TSV file.
    # Abort reading and log an error to stderr if the file is not a valid TSV
    # file, if it contains more than one row with the same id,
    # if the column name is not known, or if there are more than N_MAX rows.
    data = defaultdict(list)
    print(tsv_path)
    with open(tsv_path, 'rt', encoding="utf-8") as infp:
        reader = csv.DictReader(infp, delimiter='\t')
        for i, row in enumerate(reader):
            if i == 0:
                if set(reader.fieldnames) != set(expected_cols):
                    gotcols = ", ".join(list(set(reader.fieldnames)))
                    print(f"Invalid column names in TSV file, expected:\n  {
                          ', '.join(expected_cols)}\ngot\n  {gotcols}",
                          file=sys.stderr)
                    return None
            if i >= expected_rows:
                print(f"Too many rows in TSV file, expected {
                      expected_rows}", file=sys.stderr)
                return None
            for col_name in reader.fieldnames:
                data[col_name].append(row[col_name])
    if len(data['id']) != expected_rows:
        print(f"Missing values in TSV file, expected {
              expected_rows} rows, got {len(data['id'])}", file=sys.stderr)
        return None
    return data


def score_st1(data, targets):
    """Calculate the score for subtask 1"""
    # NOTE: targets are a dictionary with the same keys as data, and the
    # values are lists of the target values
    # for some columns where more than one prediction is allowed, the target
    # values are lists of lists

    # for those columns, where more than one prediction is allowed, we need to
    # select either the one
    # predicted by the model, or a random incorrect one if the model did not
    # predict a correct one

    check_columns(data, ST1_COLUMNS)
    check_allowed(data, 'bin_maj', ["0", "1"])
    check_allowed(data, 'bin_one', ["0", "1"])
    check_allowed(data, 'bin_all', ["0", "1"])
    check_allowed(data, 'multi_maj', MULT_LABELS)
    check_allowed(data, 'disagree_bin', ["0", "1"])

    target_bin_maj = []
    for pred, target in zip(data['bin_maj'], targets['bin_maj']):
        if isinstance(target, list):
            target_bin_maj.append(pred)
        else:
            target_bin_maj.append(target)
    targets['bin_maj'] = target_bin_maj

    target_multi_maj = []
    for pred, target in zip(data['multi_maj'], targets['multi_maj']):
        if isinstance(target, list):
            if pred not in target:
                # select a random incorrect target: just pick the first one
                target_multi_maj.append(target[0])
            else:
                # the prediction is correct
                target_multi_maj.append(pred)
        else:
            target_multi_maj.append(target)
    targets['multi_maj'] = target_multi_maj

    scores = {}
    used_scores = []
    for col_name in data.keys():
        if col_name == 'id':
            continue
        if GLOBALS['debug']:
            print(f"Calculating scores for {col_name}")
        scores[col_name +
               "_acc"] = accuracy_score(data[col_name], targets[col_name])
        scores[col_name+"_f1"] = f1_score(data[col_name],
                                          targets[col_name], average='macro')
        used_scores.append(scores[col_name+"_f1"])
    # calculate average over all f1 scores
    scores['score'] = np.mean(used_scores)
    return scores


def score_st2(data, targets):
    """Calculate the score for subtask 2"""
    check_dist(data, ['dist_bin_0', 'dist_bin_1'])
    check_dist(data, ['dist_multi_0', 'dist_multi_1',
               'dist_multi_2', 'dist_multi_3', 'dist_multi_4'])
    scores = {}
    sum_bin = 0.0
    sum_multi = 0.0
    for idx in range(len(data['id'])):
        # calculate the vectors for the binary and multi-class predictions
        dist_bin = [float(data['dist_bin_0'][idx]),
                    float(data['dist_bin_1'][idx])]
        dist_multi = [float(data[colname][idx]) for colname in [
            'dist_multi_0', 'dist_multi_1', 'dist_multi_2', 'dist_multi_3',
            'dist_multi_4']]
        # calculate the vectors for the binary and multi-class targets
        target_bin = [targets['dist_bin_0'][idx], targets['dist_bin_1'][idx]]
        target_multi = [targets['dist_multi_0'][idx],
                        targets['dist_multi_1'][idx],
                        targets['dist_multi_2'][idx],
                        targets['dist_multi_3'][idx],
                        targets['dist_multi_4'][idx]]
        # calculate the distances
        score_bin = distance.jensenshannon(dist_bin, target_bin, base=2)
        score_multi = distance.jensenshannon(dist_multi, target_multi, base=2)
        sum_bin += score_bin
        sum_multi += score_multi
    scores['js_dist_bin'] = sum_bin / len(data['id'])
    scores['js_dist_multi'] = sum_multi / len(data['id'])
    scores['score'] = np.mean([scores['js_dist_bin'], scores['js_dist_multi']])
    return scores


def main():
    # Updated paths for windows systems. Exchange for different model
    submission_dir = "../../05_results/[model]"
    submission_file = "[modelname].tsv"
    reference_dir = "../../01_data/"  # dir for the scores with gold labels
    score_dir = "../../05_results/[model]"

    parser = argparse.ArgumentParser(description='Scorer for the competition')
    parser.add_argument("--st", required=True,
                        choices=["1", "2"],
                        help='Subtask to evaluate, one of 1, 2')
    parser.add_argument(
        "--debug", help='Print debug information', action='store_true')
    args = parser.parse_args()

    GLOBALS['debug'] = args.debug
    print(f'Running scorer for subtask {args.st}')

    print("Running locally with hardcoded paths")

    targets_file = os.path.join(reference_dir, "targets.json")
    print(f"Using targets file {targets_file}")

    # Load the targets
    targets = load_targets(targets_file)
    print(f"Loaded {len(targets)} targets")

    # Index the targets for easy lookup
    targets_index = {t['id']: t for t in targets}

    # Load the submission TSV file based on subtask
    if args.st == "1":
        data = load_tsv(submission_dir, expected_rows=len(
            targets), expected_cols=ST1_COLUMNS, file=submission_file)
    elif args.st == "2":
        data = load_tsv(submission_dir, expected_rows=len(
            targets), expected_cols=ST2_COLUMNS, file=submission_file)
    else:
        print("Unknown subtask", file=sys.stderr)
        sys.exit(1)

    if data is None:
        print("Problems loading the submission, aborting", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data['id'])} rows from the submission")

    # Check if the IDs in the submission match the targets
    if set(data['id']) != set(targets_index.keys()):
        print("IDs in submission do not match IDs in targets", file=sys.stderr)
        sys.exit(1)

    # Convert targets to the same format as the submission
    targets_dir = {}
    for col_name in data.keys():
        if col_name == 'id':
            continue
        col_values = []
        for idx, id in enumerate(data['id']):
            if id not in targets_index:
                print(f"ID {id} not found in targets for id {
                      id} in row {idx}", file=sys.stderr)
                sys.exit(1)
            if col_name not in targets_index[id]:
                print(f"Column {col_name} not found in targets for id {
                      id} in row {idx}", file=sys.stderr)
                sys.exit(1)
            col_values.append(targets_index[id][col_name])
        targets_dir[col_name] = col_values

    # Score based on the subtask
    if args.st == "1":
        scores = score_st1(data, targets_dir)
    elif args.st == "2":
        scores = score_st2(data, targets_dir)
    else:
        print("Unknown subtask", file=sys.stderr)
        sys.exit(1)

    print("Scores:", scores)

    # Write the scores to a file
    with open(os.path.join(score_dir, 'scores.json'), 'w',
              encoding="utf-8") as score_file:
        score_file.write(json.dumps(scores))

    print("Ending scorer")


if __name__ == '__main__':
    main()
