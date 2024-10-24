from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import argparse
import subprocess

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="One-vs-One SVM classifier")
parser.add_argument('testfile', type=str, help="Path to the test CSV file")
args = parser.parse_args()

subprocess.run(['python', './data_preprocessing.py', args.testfile])


# Load preprocessed data
data = pd.read_csv('data/train_preprocessed.csv')

# Split the data into features and target
X = data.drop('Segmentation', axis=1)  # Replace 'Segmentation' with actual target column name
y = data['Segmentation']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get unique classes
classes = np.unique(y_train)

# Dictionary to hold one SVM per class
classifiers = {}

# Train one SVM per class (One-vs-All approach)
for cls in classes:
    # Create binary target for this class vs the rest
    binary_y_train = np.where(y_train == cls, 1, 0)
    
    # Train the SVM for this binary classification
    svm = SVC(kernel='rbf', probability=True)  # `probability=True` allows access to decision function
    svm.fit(X_train, binary_y_train)
    
    # Save the classifier
    classifiers[cls] = svm

# Make predictions
decision_values = np.zeros((X_test.shape[0], len(classes)))

# Collect decision scores for each classifier
for idx, cls in enumerate(classes):
    decision_values[:, idx] = classifiers[cls].decision_function(X_test)

# Choose the class with the highest decision score
predictions = classes[np.argmax(decision_values, axis=1)]

# Save predictions to CSV
output = pd.DataFrame({'predicted': predictions})
output.to_csv('ova.csv', index=False)

# Calculate accuracy (optional)
accuracy = accuracy_score(y_test, predictions)
print(f"One-vs-All Classifier Accuracy: {accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(y_test, predictions)
print(class_report)

# Load the test data
# test_data = pd.read_csv(args.testfile)
test_data = pd.read_csv('data/test_preprocessed.csv')

# Make predictions on the test data

decision_values = np.zeros((test_data.shape[0], len(classes)))

# Collect decision scores for each classifier
for idx, cls in enumerate(classes):
    decision_values[:, idx] = classifiers[cls].decision_function(test_data)

# Choose the class with the highest decision score
test_predictions = classes[np.argmax(decision_values, axis=1)]

# Save predictions to CSV
output = pd.DataFrame({'predicted': test_predictions})
output.to_csv('ova.csv', index=False)
