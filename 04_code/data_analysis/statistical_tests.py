# This script checks the data for normal distribution.
# Because of not enough data points, we can not conduct a ttest or a (m)anova
# Author: Niklas Donhauser
# Date: August 22, 2024

# import libraries
import pandas as pd
from scipy import stats
import sys


def read_file(file_name):
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Display the DataFrame (optional)
    print("Data Loaded:")
    print(df)
    return df


def shapiro_wilk(data, test_name):
    if test_name == 1:
        for metric in ["Score", "Multi-Maj F1", "Bin-Maj F1", "BinOne F1", "BinAll F1", "Dis. Bin F1"]:  # noqa: E501
            stat, p = stats.shapiro(data[metric])
            print("Shapiro-Wilk Test:")
            if (p > 0.05):
                print(f"Data normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501
                # Means also the data is parametric
            else:
                print(f"Data NOT normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501
    if test_name == 2:
        for metric in ["Score", "JS Dist Multi", "JS Dist Bin"]:
            stat, p = stats.shapiro(data[metric])
            print("Shapiro-Wilk Test:")
            if (p > 0.05):
                print(f"Data normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501
                # Means also the data is parametric
            else:
                print(f"Data NOT normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501
    if test_name == 3:
        for metric in ['Micro F1-Score', 'Accuracy', 'Recall', 'Precision']:
            stat, p = stats.shapiro(data[metric])
            print("Shapiro-Wilk Test:")
            if (p > 0.05):
                print(f"Data normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501
                # Means also the data is parametric
            else:
                print(f"Data NOT normal distributed: {metric}: Statistics={stat:.3f},p-value={p:.3f}")  # noqa: E501


def main():
    st1 = "../../03_input/statistical_test/ST_1.csv"
    st2 = "../../03_input/statistical_test/ST_2.csv"
    metrics = "../../03_input/statistical_test/metrics.csv"
    shapiro_wilk(read_file(st1), 1)
    shapiro_wilk(read_file(st2), 2)
    shapiro_wilk(read_file(metrics), 3)


if __name__ == '__main__':
    sys.exit(main())
