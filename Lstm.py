import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and preprocess the data
data = pd.read_csv('input.csv')

# Split data into features and target
X_numerical = data.select_dtypes(include=['float'])
X_text = data.select_dtypes(include=['object'])
y = data['target_column']

# Step 2: Text preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_text['text_column'])
X_text_seq = tokenizer.texts_to_sequences(X_text['text_column'])
X_text_padded = pad_sequences(X_text_seq, maxlen=max_len)

# Step 3: Model selection and training
# For numerical data
numerical_model = RandomForestClassifier()
numerical_model.fit(X_numerical, y)

# For text data
text_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(64),
    Dense(4, activation='softmax')
])
text_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
text_model.fit(X_text_padded, y, epochs=10, batch_size=32)

# Step 4: Evaluation
# Evaluate numerical model
numerical_preds = numerical_model.predict(X_numerical)
print("Numerical Model Classification Report:")
print(classification_report(y, numerical_preds))

# Evaluate text model
text_preds = text_model.predict_classes(X_text_padded)
print("Text Model Classification Report:")
print(classification_report(y, text_preds))

# Step 5: Deployment
# Deploy the models using your preferred method (e.g., Flask API, AWS Lambda, etc.)

# Step 6: Output CSV
# Make predictions on new data
new_data = pd.read_csv('new_data.csv')

# Preprocess new data
X_numerical_new = new_data.select_dtypes(include=['float'])
X_text_new = new_data.select_dtypes(include=['object'])
X_text_seq_new = tokenizer.texts_to_sequences(X_text_new['text_column'])
X_text_padded_new = pad_sequences(X_text_seq_new, maxlen=max_len)

# Predict using both models
numerical_preds_new = numerical_model.predict(X_numerical_new)
text_preds_new = text_model.predict_classes(X_text_padded_new)

# Store predictions in an output CSV file
output_df = pd.DataFrame({
    'Numerical_Predictions': numerical_preds_new,
    'Text_Predictions': text_preds_new
})
output_df.to_csv('output_predictions.csv', index=False)
