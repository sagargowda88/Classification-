 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Step 1: Load data
data = pd.read_csv('input.csv')

# Step 2: Handle categorical features (object type)
categorical_columns = data.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
X_encoded = encoder.fit_transform(data[categorical_columns])

# Step 3: Text preprocessing using TF-IDF
text_column_name = [col for col in data.columns if data[col].dtype == 'object'][0]  # Find the text column automatically
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_text_vectorized = vectorizer.fit_transform(data[text_column_name])

# Combine numerical, encoded categorical, and text features
X_combined = pd.concat([
    data.drop(columns=categorical_columns + [text_column_name]),
    pd.DataFrame(X_encoded.toarray()),
    pd.DataFrame(X_text_vectorized.toarray())
], axis=1)

# One-hot encode the target variable
encoder_y = OneHotEncoder(sparse=False)
y_encoded = encoder_y.fit_transform(data[['target_column']])

# Split data into features and target
X = X_combined
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
