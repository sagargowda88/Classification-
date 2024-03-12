 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Step 1: Load data
data = pd.read_csv('input.csv')

# Step 2: Split data into features and target
X = data.drop(columns=['target_column'])
y = data['target_column']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Handle categorical features (object type)
categorical_columns = X_train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Step 5: Text preprocessing using TF-IDF
text_column_name = [col for col in X_train.columns if X_train[col].dtype == 'object'][0]  # Find the text column automatically
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_text_vectorized = vectorizer.fit_transform(X_train[text_column_name])
X_test_text_vectorized = vectorizer.transform(X_test[text_column_name])

# Step 6: Combine numerical, encoded categorical, and text features
X_train_combined = pd.concat([
    X_train.drop(columns=categorical_columns + [text_column_name]),
    pd.DataFrame(X_train_encoded.toarray()),
    pd.DataFrame(X_train_text_vectorized.toarray())
], axis=1)

X_test_combined = pd.concat([
    X_test.drop(columns=categorical_columns + [text_column_name]),
    pd.DataFrame(X_test_encoded.toarray()),
    pd.DataFrame(X_test_text_vectorized.toarray())
], axis=1)

# Step 7: Train XGBoost model
model = XGBClassifier()
model.fit(X_train_combined, y_train)

# Step 8: Evaluate the model
predictions = model.predict(X_test_combined)
