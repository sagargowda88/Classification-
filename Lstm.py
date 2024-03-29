 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Create DataFrame with predictions and true labels
results_df = pd.DataFrame({'True_Label': true_labels, 'Predicted_Label': predicted_labels})

# Save predictions to CSV
results_df.to_csv('predictions.csv', index=False)

# Generate classification report
classification_rep = classification_report(true_labels, predicted_labels)
print(classification_rep)

for class_label, metrics in classification_rep.items():
    print(f'Class: {class_label}')
    print(f'Precision: {metrics["precision"]}')
    print(f'Recall: {metrics["recall"]}')
    print(f'F1-score: {metrics["f1-score"]}')
    print()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Append predictions to the original DataFrame
test_indices = X_test.index
data_test = data.loc[test_indices]
data_test['Predicted_Label'] = predicted_labels

# Save data with predictions to CSV
data_test.to_csv('output_with_predictions.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Filter failed predictions
failed_indices = X_test.index[predicted_labels != true_labels]
failed_data = data.loc[failed_indices]
failed_data['Predicted_Label'] = predicted_labels[predicted_labels != true_labels]

# Save failed predictions to CSV
failed_data.to_csv('failed_predictions.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set hyperparameters
params = {
    'max_depth': 6,
    'n_estimators': 100,
    'eta': 0.3,
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(label_encoder.classes_),  # Number of classes
    'eval_metric': 'merror'  # Multiclass error rate
}

# Train XGBoost model
model = XGBClassifier(**params)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Filter failed predictions
failed_indices = X_test.index[predicted_labels != true_labels]
failed_data = data.loc[failed_indices]
failed_data['Predicted_Label'] = predicted_labels[predicted_labels != true_labels]

# Save failed predictions to CSV
failed_data.to_csv('failed_predictions.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Generate classification report
classification_rep = classification_report(true_labels, predicted_labels, output_dict=True)

# Print the number of correct predictions for each class
print("Classification Report:")
for class_label, metrics in classification_rep.items():
    if class_label in label_encoder.classes_:  # Checking if class_label is a valid class
        total_labels = metrics['support']
        correct_predictions = metrics['precision'] * total_labels
        print(f"For class '{class_label}':")
        print(f"Correct Predictions: {correct_predictions} out of {total_labels} total labels")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom scorer for multi-label classification
def multi_label_accuracy_score(y_true, y_pred):
    num_labels = len(y_true)
    total_correct = sum((1 for i in range(num_labels) if y_true[i] == y_pred[i]))
    return total_correct / num_labels

# Parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [50, 100, 200],
}

# Create XGBoost model
xgb_model = XGBClassifier()

# Grid search with custom scorer
scorer = make_scorer(multi_label_accuracy_score)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=scorer, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train XGBoost model with best parameters
best_model = XGBClassifier(**best_params)
best_model.fit(X_train, y_train)

# Predictions
predictions = best_model.predict(X_test)

# Decode labels
predicted_labels = label_encoder.inverse_transform(predictions)
true_labels = label_encoder.inverse_transform(y_test)

# Generate classification report
classification_rep = classification_report(true_labels, predicted_labels, output_dict=True)

# Print the number of correct predictions for each class
print("Classification Report:")
for class_label, metrics in classification_rep.items():
    if class_label in label_encoder.classes_:  # Checking if class_label is a valid class
        total_labels = metrics['support']
        correct_predictions = metrics['precision'] * total_labels
        print(f"For class '{class_label}':")
        print(f"Correct Predictions: {correct_predictions} out of {total_labels} total labels")

