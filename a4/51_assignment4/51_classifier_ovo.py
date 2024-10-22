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

# Function to train binary classifiers for each pair of classes
def train_ovo_svm(X_train, y_train, classes):
    classifiers = {}
    # Train a binary SVM classifier for each pair of classes
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes[i + 1:]):
            # Get only the data points belonging to the two classes
            binary_indices = np.where((y_train == class_i) | (y_train == class_j))[0]
            X_binary = X_train.iloc[binary_indices]
            y_binary = y_train.iloc[binary_indices]
            
            # Train a binary SVM classifier
            svm = SVC(kernel='rbf')
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

# Train OvO SVM classifiers
classifiers = train_ovo_svm(X_train, y_train, classes)

# Make predictions on the test set using OvO logic
ovo_predictions = ovo_predict(X_test, classifiers, classes)

# # Save predictions to CSV
# output = pd.DataFrame({'predicted': ovo_predictions})
# output.to_csv('ovo.csv', index=False)

# Calculate accuracy
accuracy = accuracy_score(y_test, ovo_predictions)
print(f"One-vs-One Classifier Accuracy: {accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, ovo_predictions)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(y_test, ovo_predictions)
print(class_report)

# Load the test data
# test_data = pd.read_csv(args.testfile)
test_data = pd.read_csv('data/test_preprocessed.csv')

# Make predictions on the test data
test_predictions = ovo_predict(test_data, classifiers, classes)

# Save predictions to CSV
output = pd.DataFrame({'predicted': test_predictions})
output.to_csv('ovo.csv', index=False)