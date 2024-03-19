import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a pandas DataFrame
data = pd.read_csv("your_file.csv")

# One-hot encode the class variable
data = pd.get_dummies(data, columns=["class_variable"])

# Split the data into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test sets to separate CSV files
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
