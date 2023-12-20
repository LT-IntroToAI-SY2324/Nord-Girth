# breast_cancer_logistic_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    # Fetch breast cancer dataset from OpenML
    breast_cancer = fetch_openml(name="breast_cancer")

    # Create a DataFrame from the data
    data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    data['target'] = breast_cancer.target.astype(int)

    # Select one feature for simplicity (you may choose other features)
    feature = 'mean area'
    X = data[[feature]]
    y = data['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and display confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualize the logistic regression curve
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='black', label='True values')
    plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted values')
    plt.plot(X_test, model.predict_proba(X_test)[:, 1], color='blue', linewidth=3, label='Logistic Regression Curve')
    plt.xlabel(feature)
    plt.ylabel('Target (Malignant: 0, Benign: 1)')
    plt.legend()
    plt.title('Logistic Regression on Breast Cancer Data')
    plt.show()

if __name__ == "__main__":
    main()