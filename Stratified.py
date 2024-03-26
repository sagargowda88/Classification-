import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

# Load data
data = pd.read_csv('input.csv')

# Handle categorical features (object type) with label encoding
categorical_columns = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
data[categorical_columns] = data[categorical_columns].apply(encoder.fit_transform)

# Text preprocessing using TF-IDF for text columns
text_column_name = [col for col in data.columns if data[col].dtype == 'object'][0]  # Find the text column automatically
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_text_vectorized = vectorizer.fit_transform(data[text_column_name])

# Combine numerical, encoded categorical, and text features
X_combined = pd.concat([
    data.drop(columns=[text_column_name]),
    pd.DataFrame(X_text_vectorized.toarray())
], axis=1)

# Encode the target variable using label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['target_column'])

# Split data into features and target
X = X_combined
y = y_encoded

# Perform stratified sampling with 5 splits to split data into training and testing sets
stratified_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Lists to store accuracy scores and classification reports for each split
accuracy_scores = []
classification_reports = []

# Iterate over each split
for train_index, test_index in stratified_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train XGBoost model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)

    # Generate classification report
    report = classification_report(y_test, predictions, output_dict=True)
    classification_reports.append(report)

# Calculate and print the final accuracy
final_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"Final Accuracy: {final_accuracy}")

# Calculate and print the final classification report
final_classification_report = classification_report(
    y_true=y,
    y_pred=predictions,
    output_dict=False  # Set to True if you want the report as a dictionary
)
print("Final Classification Report:")
print(final_classification_report)
