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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="One-vs-All SVM classifier")
parser.add_argument('testfile', type=str, help="Path to the test CSV file")
args = parser.parse_args()

subprocess.run(['python', './data_preprocessing.py', args.testfile])

data = pd.read_csv('data/train_preprocessed.csv')

x = data.drop('Segmentation', axis=1)  
y = data['Segmentation']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
classes = np.unique(ytrain)

# hyperparameter ranges for C and gamma
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

best_accuracy = 0
best_params = None

for C in C_values:
    for gamma in gamma_values:
        print(f"Training with C={C}, gamma={gamma}")
        classifiers = {}
        # Train one SVM per class (One-vs-All approach)
        for cls in classes:
            binary_y_train = np.where(ytrain == cls, 1, 0)

            svm = SVC(kernel='rbf', probability=True, C=C, gamma=gamma)
            svm.fit(xtrain, binary_y_train)

            classifiers[cls] = svm

        decision_vals = np.zeros((xtest.shape[0], len(classes)))

        # Collecting decision scores for each classifier
        for idx, cls in enumerate(classes):
            decision_vals[:, idx] = classifiers[cls].decision_function(xtest)

        # Choosing the class with the highest decision score
        predictions = classes[np.argmax(decision_vals, axis=1)]

        accuracy = accuracy_score(ytest, predictions)
        print(f"Accuracy with C={C}, gamma={gamma}: {accuracy:.2f}")
        
        # Keeping track of the best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (C, gamma)

print(f"Best Accuracy: {best_accuracy:.2f} with C={best_params[0]}, gamma={best_params[1]}")

print("Using best parameters for final prediction")
classifiers = {}

for cls in classes:
    binary_y_train = np.where(ytrain == cls, 1, 0)
    svm = SVC(kernel='rbf', probability=True, C=best_params[0], gamma=best_params[1])
    svm.fit(xtrain, binary_y_train)
    classifiers[cls] = svm

test_data = pd.read_csv('data/test_preprocessed.csv')

decision_vals = np.zeros((test_data.shape[0], len(classes)))

for idx, cls in enumerate(classes):
    decision_vals[:, idx] = classifiers[cls].decision_function(test_data)

test_pred = classes[np.argmax(decision_vals, axis=1)]

test_pred = pd.Series(test_pred).map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})

output = pd.DataFrame({'predicted': test_pred})
output.to_csv('ova.csv', index=False)

decision_vals = np.zeros((xtest.shape[0], len(classes)))

for idx, cls in enumerate(classes):
    decision_vals[:, idx] = classifiers[cls].decision_function(xtest)

final_pred = classes[np.argmax(decision_vals, axis=1)]

final_accuracy = accuracy_score(ytest, final_pred)
print(f"Final One-vs-All Classifier Accuracy: {final_accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(ytest, final_pred)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(ytest, final_pred)
print(class_report)

# Plotting the confusion matrix with proper class names
class_names = ['A', 'B', 'C', 'D']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for One-vs-All SVM Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

