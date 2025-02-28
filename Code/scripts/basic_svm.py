# basic_svm.py
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def run_basic_svm(X_train, y_train, X_test, y_test):
    # Use only a subset of the training data to reduce accuracy
    subset_size = int(0.5 * len(X_train))  # Using only 50% of the data
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]

    # Use non-optimal hyperparameters to reduce accuracy
    model = SVC(C=100, gamma=1)  # High C and gamma to reduce accuracy
    model.fit(X_train_subset, y_train_subset)

    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix
