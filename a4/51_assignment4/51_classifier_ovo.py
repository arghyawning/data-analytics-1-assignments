import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter
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

data = pd.read_csv('data/train_preprocessed.csv')

X = data.drop('Segmentation', axis=1)
y = data['Segmentation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classes = np.unique(y_train)

# Function to train binary classifiers for each pair of classes with given hyperparameters
def train_ovo_svm(X_train, y_train, classes, C, gamma):
    classifiers = {}
    # Train a binary SVM classifier for each pair of classes
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes[i + 1:]):
            # Get only the data points belonging to the two classes
            binary_indices = np.where((y_train == class_i) | (y_train == class_j))[0]
            X_binary = X_train.iloc[binary_indices]
            y_binary = y_train.iloc[binary_indices]
            
            # Train a binary SVM classifier with specified C and gamma
            svm = SVC(kernel='rbf', C=C, gamma=gamma)
            svm.fit(X_binary, y_binary)
            classifiers[(class_i, class_j)] = svm
    return classifiers

# Function to predict using the trained OvO classifiers
def ovo_predict(X_test, classifiers, classes):
    predictions = []
    for x in X_test.values:
        votes = []
        # For each pair of classes, make a prediction
        for (class_i, class_j), svm in classifiers.items():
            pred = svm.predict([x])[0]
            votes.append(pred)
        
        # Majority voting to decide the final class
        most_common_class = Counter(votes).most_common(1)[0][0]
        predictions.append(most_common_class)
    return predictions

# Define hyperparameter ranges for C and gamma
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

# Iterate over combinations of C and gamma
best_accuracy = 0
best_params = None

for C in C_values:
    for gamma in gamma_values:
        print(f"Training with C={C}, gamma={gamma}")
        # Train OvO SVM classifiers with current C and gamma
        classifiers = train_ovo_svm(X_train, y_train, classes, C, gamma)
        
        # Make predictions on the test set using OvO logic
        ovo_predictions = ovo_predict(X_test, classifiers, classes)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, ovo_predictions)
        print(f"Accuracy with C={C}, gamma={gamma}: {accuracy:.2f}")
        
        # Keep track of the best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (C, gamma)

# Print best hyperparameters and their accuracy
print(f"Best Accuracy: {best_accuracy:.2f} with C={best_params[0]}, gamma={best_params[1]}")

# Make final predictions on the test data with the best parameters
print("Using best parameters for final prediction")
classifiers = train_ovo_svm(X_train, y_train, classes, best_params[0], best_params[1])
test_data = pd.read_csv('data/test_preprocessed.csv')
test_predictions = ovo_predict(test_data, classifiers, classes)

# map the predictions to the original classes (A, B, C, D)
test_predictions = pd.Series(test_predictions).map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})

# Save final predictions to CSV
output = pd.DataFrame({'predicted': test_predictions})
output.to_csv('ovo.csv', index=False)

# Evaluate final model performance on test data
ovo_predictions = ovo_predict(X_test, classifiers, classes)
accuracy = accuracy_score(y_test, ovo_predictions)
print(f"Final One-vs-One Classifier Accuracy: {accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, ovo_predictions)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(y_test, ovo_predictions)
print(class_report)
