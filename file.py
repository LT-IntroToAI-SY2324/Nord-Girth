# breast_cancer_logistic_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    try:
        # Load breast cancer dataset from CSV file
        breast_cancer_data = pd.read_csv("breast_cancer_dataset.csv")

        # Select features and target from the breast cancer dataset
        # Adjust column names based on your actual dataset
        features = breast_cancer_data.columns[:-1]  # Assuming the last column is the target
        X = breast_cancer_data[features]
        y = breast_cancer_data['target']

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

        # Visualize the logistic regression curve for one feature for simplicity
        feature = features[0]
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[feature], y_test, color='black', label='True values')
        plt.scatter(X_test[feature], y_pred, color='red', marker='x', label='Predicted values')
        plt.plot(X_test[feature], model.predict_proba(X_test)[:, 1], color='blue', linewidth=3, label='Logistic Regression Curve')
        plt.xlabel(feature)
        plt.ylabel('Target (Malignant: 0, Benign: 1)')
        plt.legend()
        plt.title('Logistic Regression on Breast Cancer Data')
        plt.show()

    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
    except pd.errors.ParserError:
        print("Error: Unable to parse the CSV file. Check the file format and delimiter.")

if __name__ == "__main__":
    main()