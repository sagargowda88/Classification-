# csa.py
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

class CSA:
    def __init__(self, num_iters=5, num_XGB_models=20, confidence_choice="ttest", verbose=False):
        """
        Initialize Confident Sinkhorn Allocation (CSA) algorithm.

        Args:
            num_iters (int): Number of pseudo-iterations.
            num_XGB_models (int): Number of XGBoost models to use.
            confidence_choice (str): Confidence choice for CSA algorithm.
            verbose (bool): Verbosity level.
        """
        self.num_iters = num_iters
        self.num_XGB_models = num_XGB_models
        self.confidence_choice = confidence_choice
        self.verbose = verbose

        self.XGBmodels_list = [XGBClassifier() for _ in range(num_XGB_models)]
        self.test_acc = []

    def fit(self, X_train, y_train):
        """
        Fit CSA algorithm.

        Args:
            X_train (numpy array): Training features.
            y_train (numpy array): Training labels.
        """
        for _ in range(self.num_iters):
            for model in self.XGBmodels_list:
                model.fit(X_train, y_train)
            # Evaluate performance
            self.test_acc.append(accuracy_score(y_train, self.predict(X_train)))

    def predict(self, X):
        """
        Predict labels for input data.

        Args:
            X (numpy array): Input data.

        Returns:
            numpy array: Predicted labels.
        """
        # Compute average prediction of XGBoost models
        preds = np.mean([model.predict(X) for model in self.XGBmodels_list], axis=0)
        # Threshold predictions
        labels = np.where(preds > 0.5, 1, 0)
        return labels

# experiment.py
import os
import argparse
import logging
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utilities.utils import load_data, save_labels_to_csv

def run_experiments(args):
    """
    Run experiments using CSA algorithm.

    Args:
        args (argparse.Namespace): Arguments for running experiments.
    """
    # Load data and shuffle
    data = load_data(args.csv_file)
    data_shuffled = data.sample(frac=1, random_state=42)

    # Split data into train and test sets
    X = data_shuffled.drop(columns=['target'])  # Assuming 'target' column contains labels
    y = data_shuffled['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Initialize CSA algorithm
    csa = CSA(num_iters=args.numIters, num_XGB_models=args.numXGBs,
              confidence_choice=args.confidence_choice, verbose=args.verbose)

    # Fit CSA algorithm
    csa.fit(X_train.values, y_train.values)

    # Predict labels for the test data
    predicted_labels = csa.predict(X_test.values)

    # Compute prediction accuracy
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"Prediction Accuracy: {accuracy:.2f}")

    # Save data with predicted labels to CSV file
    save_labels_to_csv(X_test, predicted_labels, args.output_csv)

def main(args):
    """
    Main function to run experiments.

    Args:
        args (argparse.Namespace): Arguments for running experiments.
    """
    # Run experiments
    run_experiments(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for CSA experiments')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('output_csv', type=str, help='Path to save the CSV file with predicted labels')
    parser.add_argument('--numIters', type=int, default=5, help='Number of pseudo iterations')
    parser.add_argument('--numXGBs', type=int, default=10, help='Number of XGB models')
    parser.add_argument('--confidence_choice', type=str, default='ttest', help='Confidence choice for CSA algorithm')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    args = parser.parse_args()
    main(args)


# python experiment.py input.csv output.csv --numIters 5 --numXGBs 10 --confidence_choice ttest --verbose
