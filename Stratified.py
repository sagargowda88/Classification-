 import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

# Load data from CSV
data = pd.read_csv("data.csv")

# Split features and target variable
X_text = data["text_column"]
X_categorical = data["categorical_column"]
y = data["target"]

# Label encode categorical features
label_encoder = LabelEncoder()
X_categorical_encoded = label_encoder.fit_transform(X_categorical)

# TF-IDF vectorization for text features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)

# Combine text TF-IDF and categorical features
import scipy.sparse as sp
X_combined = sp.hstack((X_text_tfidf, X_categorical_encoded.reshape(-1, 1)))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Calculate class weights based on the encoded labels
class_weights = class_weight.compute_class_weight('balanced', classes=y.unique(), y=y_train)

# Define XGBoost parameters with scale_pos_weight
params = {
    'objective': 'multi:softmax',
    'num_class': 4,
    'eval_metric': 'merror',
    'scale_pos_weight': class_weights.tolist()
}

# Convert data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train XGBoost model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Make predictions
preds = bst.predict(dtest)

# Evaluate model
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, preds))
