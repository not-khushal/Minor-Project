# scripts/utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to load the MNIST dataset
def load_data():
    train_data = pd.read_csv('data/mnist_train.csv')
    X = train_data.drop('label', axis=1).values
    y = train_data['label'].values
    return X, y

# Function to standardize the dataset
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
