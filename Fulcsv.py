 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
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

# Get indices of rows present in train but not in test
train_indices = set(X_train.index)
test_indices = set(X_test.index)
indices_not_in_test = list(train_indices - test_indices)

# Filter data to include only rows present in train but not in test
rows_not_in_test = data.loc[indices_not_in_test]

# Save the filtered data to a CSV file
rows_not_in_test.to_csv('rows_not_in_test.csv', index=False)
