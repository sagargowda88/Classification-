 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Load data
data = pd.read_csv('input.csv')

# Handle categorical features (object type) with label encoding
categorical_columns = data.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
data[categorical_columns] = data[categorical_columns].apply(encoder.fit_transform)

# Use Sentence Transformer for text encoding
text_column_name = [col for col in data.columns if data[col].dtype == 'object'][0]  # Find the text column automatically
model = SentenceTransformer('distilbert-base-nli-mean-tokens')  # Choose your desired pre-trained model
X_text_vectorized = model.encode(data[text_column_name].tolist())

# Combine numerical, encoded categorical, and text features
X_combined = pd.concat([
    data.drop(columns=[text_column_name]),
    pd.DataFrame(X_text_vectorized)
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
