import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):
        """
        Given a list of features vectors of testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(self.trainingFeatures, self.trainingLabels)
        return classifier.predict(features)


def load_data(filename):
    """
    Load spam data from a CSV file filename and convert into a list of
    feature vectors and a list of target labels. Return a tuple (features, labels).

    Feature vectors should be a list of lists, where each list contains the
    57 feature values.

    Labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    features = []
    labels = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            features.append(list(map(float, row[:-1])))
            labels.append(int(row[-1]))
    return np.array(features), np.array(labels)


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    return (features - means) / stds


def train_mlp_model(features, labels):
    """
    Given a list of feature vectors and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    model = MLPClassifier(max_iter=2000)  # Set max_iter to a higher value
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


def main():
    # Check command-line arguments
    filename = "./spambase.csv"  # Replace with the actual path to your CSV file

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(filename)
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions_nn = model_nn.predict(X_test, K)
    accuracy_nn, precision_nn, recall_nn, f1_nn = evaluate(y_test, predictions_nn)

    # Print results
    print("* 1-Nearest Neighbor Results *")
    print("Accuracy: ", accuracy_nn)
    print("Precision: ", precision_nn)
    print("Recall: ", recall_nn)
    print("F1: ", f1_nn)

    # Train an MLP model and make predictions
    model_mlp = train_mlp_model(X_train, y_train)
    predictions_mlp = model_mlp.predict(X_test)
    accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = evaluate(y_test, predictions_mlp)

    # Print results for MLP
    print("\n* MLP Results *")
    print("Accuracy: ", accuracy_mlp)
    print("Precision: ", precision_mlp)
    print("Recall: ", recall_mlp)
    print("F1: ", f1_mlp)

    # Confusion Matrix for 1-Nearest Neighbor
    cm_nn = confusion_matrix(y_test, predictions_nn)
    print("\nConfusion Matrix for 1-Nearest Neighbor:")
    print(cm_nn)

    # Confusion Matrix for MLP
    cm_mlp = confusion_matrix(y_test, predictions_mlp)
    print("\nConfusion Matrix for MLP:")
    print(cm_mlp)


if __name__ == "__main__":
    main()