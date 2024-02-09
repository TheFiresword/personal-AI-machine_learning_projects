import csv
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidences, labels = [], []
    import sys
    path = sys.executable
    path = path[:path.index("shopping")+len("shopping")]
    path = os.path.join(path, filename)

    month_dict = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    with open(path, newline='') as csvfile:
        result = csv.reader(csvfile)
        count = 0
        for row in result:
            if count > 0:
                new_evidence = row[:-1]
                new_label = 1 if row[-1] == 'TRUE' else 0
                for i in range(10, 17):
                    if i == 10:
                        new_evidence[i] = month_dict[new_evidence[i]]
                    elif i == 15:
                        if new_evidence[i] == "Returning_Visitor":
                            new_evidence[i] = 1
                        else:
                            new_evidence[i] = 0
                    elif i == 16:
                        new_evidence[i] = 1 if new_evidence[i] is True else 0
                    else:
                        new_evidence[i] = int(new_evidence[i])
                new_evidence[0] = int(new_evidence[0])
                new_evidence[2] = int(new_evidence[2])
                new_evidence[4] = int(new_evidence[4])

                for i in range(5, 10):
                    new_evidence[i] = float(new_evidence[i])
                new_evidence[3] = float(new_evidence[3])
                new_evidence[1] = float(new_evidence[1])

                evidences.append(new_evidence)
                labels.append(new_label)
            count += 1

    return evidences, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    return KNeighborsClassifier(n_neighbors=1).fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity, specificity = 0, 0
    true_labels = [i for i in range(len(labels)) if labels[i] == 1]
    assert len(true_labels) >= 1
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            if i in true_labels:
                sensitivity += 1
            else:
                specificity += 1
    sensitivity /= len(true_labels)
    specificity /= (len(labels)-len(true_labels))
    return sensitivity, specificity


if __name__ == "__main__":
    main()
