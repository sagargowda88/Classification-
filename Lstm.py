 import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load data
data = pd.read_csv('input.csv')

# Step 2: Split data into train and test sets
X = data.drop(columns=['target_column'])
y = data['target_column']

# One-hot encode the target variable
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 3: Text preprocessing
text_column_name = [col for col in X_train.columns if X_train[col].dtype == 'object'][0]  # Find the text column automatically
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train[text_column_name])
X_train_text_seq = tokenizer.texts_to_sequences(X_train[text_column_name])
X_test_text_seq = tokenizer.texts_to_sequences(X_test[text_column_name])
max_len = 100  # Set the maximum length of sequences
X_train_text_padded = pad_sequences(X_train_text_seq, maxlen=max_len)
X_test_text_padded = pad_sequences(X_test_text_seq, maxlen=max_len)

# Step 4: Vectorize text data
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_text_vectorized = vectorizer.fit_transform(X_train[text_column_name])
X_test_text_vectorized = vectorizer.transform(X_test[text_column_name])

# Step 5: Combine numerical and text features
X_train_combined = pd.concat([X_train.drop(columns=[text_column_name]), pd.DataFrame(X_train_text_vectorized.toarray())], axis=1)
X_test_combined = pd.concat([X_test.drop(columns=[text_column_name]), pd.DataFrame(X_test_text_vectorized.toarray())], axis=1)

# Step 6: Model training
model = XGBClassifier()
model.fit(X_train_combined, y_train)

# Step 7: Evaluation
# Evaluate model
predictions = model.predict(X_test_combined)
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 8: Output CSV
# Store predictions in an output CSV file
output_df = pd.DataFrame({
    'Predictions': predictions.argmax(axis=1),
    'Actual_Labels': y_test.argmax(axis=1)
})
output_df.to_csv('output_predictions.csv', index=False)
