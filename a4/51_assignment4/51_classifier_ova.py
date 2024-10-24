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

# Define hyperparameter ranges for C and gamma
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

best_accuracy = 0
best_params = None

# Iterate over combinations of C and gamma
for C in C_values:
    for gamma in gamma_values:
        print(f"Training with C={C}, gamma={gamma}")
        
        classifiers = {}

        # Train one SVM per class (One-vs-All approach)
        for cls in classes:
            # Create binary target for this class vs the rest
            binary_y_train = np.where(y_train == cls, 1, 0)

            # Train the SVM for this binary classification with specified C and gamma
            svm = SVC(kernel='rbf', probability=True, C=C, gamma=gamma)
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

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy with C={C}, gamma={gamma}: {accuracy:.2f}")
        
        # Keep track of the best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (C, gamma)

# Print best hyperparameters and their accuracy
print(f"Best Accuracy: {best_accuracy:.2f} with C={best_params[0]}, gamma={best_params[1]}")

# Make final predictions on the test data with the best parameters
print("Using best parameters for final prediction")
classifiers = {}

for cls in classes:
    binary_y_train = np.where(y_train == cls, 1, 0)
    svm = SVC(kernel='rbf', probability=True, C=best_params[0], gamma=best_params[1])
    svm.fit(X_train, binary_y_train)
    classifiers[cls] = svm

# Load the test data
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

# Evaluate final model performance on the test set
decision_values = np.zeros((X_test.shape[0], len(classes)))

# Collect decision scores for each classifier on the test set
for idx, cls in enumerate(classes):
    decision_values[:, idx] = classifiers[cls].decision_function(X_test)

# Choose the class with the highest decision score
final_predictions = classes[np.argmax(decision_values, axis=1)]

# Calculate final accuracy
final_accuracy = accuracy_score(y_test, final_predictions)
print(f"Final One-vs-All Classifier Accuracy: {final_accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, final_predictions)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(y_test, final_predictions)
print(class_report)
