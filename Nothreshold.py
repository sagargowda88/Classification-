import csv
import numpy as np

def run_test(args):
    # Existing code here...

    # Convert probabilities to binary labels and write to CSV
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class1', 'Class2', 'Class3', 'Class4'])  # Replace with your actual class names
        for result in results:
            binary_result = np.zeros(len(result), dtype=int)  # Initialize binary labels with zeros
            max_index = np.argmax(result)  # Find index of maximum probability
            binary_result[max_index] = 1  # Set maximum probability to 1
            writer.writerow(binary_result)
