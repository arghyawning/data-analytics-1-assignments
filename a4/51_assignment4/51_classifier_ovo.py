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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Parser for input file
parser = argparse.ArgumentParser(description="One-vs-One SVM classifier")
parser.add_argument('testfile', type=str, help="Path to the test CSV file")
args = parser.parse_args()

subprocess.run(['python', './data_preprocessing.py', args.testfile])

data = pd.read_csv('data/train_preprocessed.csv')

x = data.drop(columns='Segmentation')
y = data['Segmentation']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

classes = np.unique(ytrain)

def train_ovo_svm(xtrain, ytrain, classes, C, gamma):
    classifiers = {}
    # Training a binary SVM classifier for each pair of classes
    for i, class_i in enumerate(classes):
        for class_j in classes[i + 1:]:
            # Getting only the data points belonging to the two classes
            common_points = np.where((ytrain == class_i) | (ytrain == class_j))[0]
            xbinary = xtrain.iloc[common_points]
            ybinary = ytrain.iloc[common_points]
            
            # Train a binary SVM classifier with specified C and gamma
            svm = SVC(kernel='rbf', C=C, gamma=gamma)
            svm.fit(xbinary, ybinary)
            classifiers[(class_i, class_j)] = svm
    return classifiers

def ovo_predict(xtest, classifiers, classes):
    predictions = []
    for x in xtest.values:
        votes = []
        for (class_i, class_j), svm in classifiers.items():
            pred = svm.predict([x])[0]
            votes.append(pred)
        
        # Majority voting to decide the final class
        majority_class = Counter(votes).most_common(1)[0][0]
        predictions.append(majority_class)
    return predictions

# hyperparameter ranges for C and gamma
C_values = [0.1, 1, 10]
gamma_values = ['scale', 'auto', 0.01, 0.1, 1]

best_accuracy = 0
best_params = None

for C in C_values:
    for gamma in gamma_values:
        print(f"Training with C={C}, gamma={gamma}")
        classifiers = train_ovo_svm(xtrain, ytrain, classes, C, gamma)
        ovo_pred = ovo_predict(xtest, classifiers, classes)
        
        accuracy = accuracy_score(ytest, ovo_pred)
        print(f"Accuracy with C={C}, gamma={gamma}: {accuracy:.2f}")
        
        # Keeping track of the best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (C, gamma)

print(f"Best Accuracy: {best_accuracy:.2f} with C={best_params[0]}, gamma={best_params[1]}")

# Making final predictions on the test data with the best parameters
print("Using best parameters for final prediction")
classifiers = train_ovo_svm(xtrain, ytrain, classes, best_params[0], best_params[1])
test_data = pd.read_csv('data/test_preprocessed.csv')
test_pred = ovo_predict(test_data, classifiers, classes)

# map the predictions to the original classes (A, B, C, D)
test_pred = pd.Series(test_pred).map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})

output = pd.DataFrame({'predicted': test_pred})
output.to_csv('ovo.csv', index=False)

ovo_pred = ovo_predict(xtest, classifiers, classes)
accuracy = accuracy_score(ytest, ovo_pred)
print(f"Final One-vs-One Classifier Accuracy: {accuracy:.2f}")

print("Confusion Matrix")
conf_matrix = confusion_matrix(ytest, ovo_pred)
print(conf_matrix)

print("Classification Report")
class_report = classification_report(ytest, ovo_pred)
print(class_report)

# plotting the confusion matrix 
class_names = ['A', 'B', 'C', 'D']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for One-vs-One SVM Classifier')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
