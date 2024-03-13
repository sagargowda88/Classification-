import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the CSV file
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
    pd.DataFrame(X_text_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
], axis=1)

# Target variable
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize the XGBClassifier
classifier = XGBClassifier()

# Initialize the MultiOutputClassifier
multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)

# Train the model
multi_target_classifier.fit(X_train, y_train)

# Predict on the test set
predictions = multi_target_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions))

# Example prediction
new_data = pd.DataFrame({
    'numerical_feature1': [1],
    'numerical_feature2': [2],
    'categorical_feature': ['new_categorical_value'],
    'text_feature': ['new attribute text']
})
new_data[categorical_columns] = new_data[categorical_columns].apply(encoder.transform)
new_text_vectorized = vectorizer.transform(new_data[text_column_name])
new_X_combined = pd.concat([
    new_data.drop(columns=[text_column_name]),
    pd.DataFrame(new_text_vectorized.toarray(), columns=vectorizer.get_feature_names_out())
], axis=1)
new_prediction = multi_target_classifier.predict(new_X_combined)
print("Prediction for new instance:", new_prediction)
