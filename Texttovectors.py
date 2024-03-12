 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from algorithm.csa import CSA
from sentence_transformers import SentenceTransformer

def save_labels_to_csv(data, labels, output_file):
    """
    Save data with predicted labels to a CSV file.

    Args:
        data (pandas DataFrame): Input data.
        labels (numpy array): Predicted labels.
        output_file (str): Path to save the CSV file.
    """
    data_with_labels = data.copy()
    data_with_labels['predicted_label'] = labels
    data_with_labels.to_csv(output_file, index=False)

def run_experiments(args):
    """
    Run experiments using CSA algorithm.

    Args:
        args (argparse.Namespace): Arguments for running experiments.
    """
    # Load CSV data and shuffle
    data = pd.read_csv(args.csv_file)
    data_shuffled = data.sample(frac=1, random_state=42)

    # Encode target class into numerical labels
    label_encoder = LabelEncoder()
    data_shuffled['label'] = label_encoder.fit_transform(data_shuffled['class_title'])

    # Split data into train and test sets
    X = data_shuffled.drop(columns=['class_title', 'label'])
    y = data_shuffled['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    # Initialize SentenceTransformer model
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Convert each text column to vectors
    X_train_text_embeddings = []
    X_test_text_embeddings = []
    for column in X_train.select_dtypes(include='object').columns:
        train_embeddings = sentence_model.encode(X_train[column].tolist())
        test_embeddings = sentence_model.encode(X_test[column].tolist())
        X_train_text_embeddings.append(train_embeddings)
        X_test_text_embeddings.append(test_embeddings)

    # Concatenate text embeddings with other features
    X_train_final = X_train.drop(columns=X_train.select_dtypes(include='object').columns).values
    X_test_final = X_test.drop(columns=X_test.select_dtypes(include='object').columns).values
    for train_embedding, test_embedding in zip(X_train_text_embeddings, X_test_text_embeddings):
        X_train_final = np.concatenate([X_train_final, train_embedding], axis=1)
        X_test_final = np.concatenate([X_test_final, test_embedding], axis=1)

    # Initialize CSA algorithm
    csa = CSA(num_iters=args.numIters, num_XGB_models=args.numXGBs,
              confidence_choice=args.confidence_choice, verbose=args.verbose)

    # Fit CSA algorithm
    csa.fit(X_train_final, y_train.values)

    # Predict labels for the test data
    predicted_labels = csa.predict(X_test_final)

    # Compute prediction accuracy
    accuracy = (predicted_labels == y_test).mean()
    print(f"Prediction Accuracy: {accuracy:.2f}")

    # Save data with predicted labels to CSV file
    save_labels_to_csv(X_test, predicted_labels, args.output_csv)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Args for CSA experiments')
    parser.add_argument('--numIters', type=int, default=5, help='number of Pseudo Iterations')
    parser.add_argument('--numXGBs', type=int, default=10, help='number of XGB models, M=?')
    parser.add_argument('--confidence_choice', type=str, default='ttest', help='confidence choices: ttest | variance | entropy | None')
    parser.add_argument('--csv_file', type=str, default='your_data.csv', help='path to input CSV file')
    parser.add_argument('--output_csv', type=str, default='predicted_labels.csv', help='name of output CSV file')
    parser.add_argument('--verbose', action='store_true', help='verbose True or False')

    args = parser.parse_args()
