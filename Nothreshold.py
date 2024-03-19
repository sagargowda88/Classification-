import csv

def run_test(args):
    # Existing code here...

    # Store predictions as binary labels
    binary_predictions = []

    # Convert probabilities to binary labels
    binary_results = [[1 if i == result.index(max(result)) else 0 for i in range(len(result))] for result in result]

    # Write binary predictions to CSV
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class1', 'Class2', 'Class3', 'Class4'])  # Replace with your actual class names
        writer.writerows(binary_results)
