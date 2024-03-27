import pandas as pd

from classifier.data_loader import id2label

# Function to classify a list of texts using the model
def classify(texts_to_classify):
    # Tokenize all texts in the batch
    inputs = tokenizer(
        texts_to_classify,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Process the outputs to get the probability distribution
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top class and the corresponding probability (certainty) for each input text
    confidences, predicted_classes = torch.max(probs, dim=1)
    predicted_classes = (
        predicted_classes.cpu().numpy()
    )  # Move to CPU for numpy conversion if needed
    confidences = confidences.cpu().numpy()  # Same here

    # Map predicted class IDs to labels
    predicted_labels = [id2label[class_id] for class_id in predicted_classes]

    # Zip together the predicted labels and confidences and convert to a list of tuples
    return list(zip(predicted_labels, confidences))

# Load the CSV file with text data
df = pd.read_csv("your_csv_file.csv")

# Predict labels for the text column
predictions = classify(df["text"].tolist())

# Add the predicted labels to the DataFrame
df["predicted_label"] = [pair[0] for pair in predictions]
df["confidence"] = [pair[1] for pair in predictions]

# Save the modified DataFrame to a new CSV file
df.to_csv("predicted_labels.csv", index=False)
