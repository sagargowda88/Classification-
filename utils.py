 import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_train_test_unlabeled(_datasetName, path_to_data, random_state=0):
    """
    Load data from CSV and split into train, test, and unlabeled sets.
    """
    # Load CSV
    df = pd.read_csv(path_to_data)
    
    # Extract features and labels
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    
    # Split data into train, test, and unlabeled sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, test_size=0.5, random_state=random_state)
    
    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_unlabeled = scaler.transform(x_unlabeled)
    
    return x_train, y_train, x_test, y_test, x_unlabeled

def get_train_test_unlabeled_for_multilabel(_datasetName, path_to_data, random_state=0):
    """
    Load multilabel data from CSV and split into train, test, and unlabeled sets.
    """
    # Load CSV
    df = pd.read_csv(path_to_data)
    
    # Extract features and labels
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    
    # Split data into train, test, and unlabeled sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    x_train, x_unlabeled, y_train, y_unlabeled = train_test_split(x_train, y_train, test_size=0.5, random_state=random_state)
    
    return x_train, y_train, x_test, y_test, x_unlabeled

# Usage example
x_train, y_train, x_test, y_test, x_unlabeled = get_train_test_unlabeled("dataset_name", "path_to_data.csv")
