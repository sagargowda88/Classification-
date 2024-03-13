import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Example values, adjust as needed
    'max_depth': [3, 5, 7],            # Example values, adjust as needed
    'learning_rate': [0.1, 0.01]       # Example values, adjust as needed
}

# Initialize XGBoost classifier
xgb = XGBClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Train the model with the best parameters
best_xgb = XGBClassifier(**best_params)
best_xgb.fit(X_train, y_train)

# Predictions
best_predictions = best_xgb.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, best_predictions))

# Decode labels
best_predicted_labels = label_encoder.inverse_transform(best_predictions)

# Filter failed predictions
best_failed_indices = X_test.index[best_predicted_labels != true_labels]
best_failed_data = data.loc[best_failed_indices]
best_failed_data['Predicted_Label'] = best_predicted_labels[best_predicted_labels != true_labels]

# Save failed predictions to CSV
best_failed_data.to_csv('best_failed_predictions.csv', index=False)
