"""
Two-Stage Machine Learning Pipeline
Stage 1: Multiple Linear Regression to predict BloodPressure
Stage 2: KNN Classification to predict Outcome
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)


def load_data():
    """Load training and test datasets."""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


def stage1_regression(train_df, test_df):
    """
    Stage 1: Train Multiple Linear Regression to predict BloodPressure.
    Replace BloodPressure in test dataset with predictions.
    """
    # Prepare features: all columns except BloodPressure and Outcome
    feature_cols = [col for col in train_df.columns
                    if col not in ['BloodPressure', 'Outcome']]

    # Training data
    X_train = train_df[feature_cols]
    y_train = train_df['BloodPressure']

    # Train the regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Predict BloodPressure for test data
    X_test = test_df[feature_cols]
    blood_pressure_predictions = reg_model.predict(X_test)

    # Create a copy of test dataset and replace BloodPressure column
    test_df_modified = test_df.copy()
    test_df_modified['BloodPressure'] = blood_pressure_predictions

    return test_df_modified, reg_model


def stage2_knn_classification(train_df, test_df_modified):
    """
    Stage 2: Train KNN classifiers with k=1 to 19.
    Evaluate each model and return accuracy results.
    """
    # Prepare features: all columns except Outcome
    feature_cols = [col for col in train_df.columns
                    if col != 'Outcome']

    # Training data
    X_train = train_df[feature_cols]
    y_train = train_df['Outcome']

    # Test data (with predicted BloodPressure)
    X_test = test_df_modified[feature_cols]
    y_test = test_df_modified['Outcome']

    # Dictionary to store k -> accuracy mapping
    accuracy_results = {}

    # Train and evaluate KNN models for k=1 to 19
    for k in range(1, 20):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)

        # Predict on test data
        y_pred = knn_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results[k] = accuracy

    return accuracy_results


def print_results(accuracy_results):
    """Print results in a clear format."""
    print("\n" + "="*50)
    print("KNN Classification Results (k -> accuracy)")
    print("="*50)

    # Print table
    print(f"\n{'k':<5} {'Accuracy':<15}")
    print("-" * 20)
    for k in sorted(accuracy_results.keys()):
        print(f"{k:<5} {accuracy_results[k]:<15.6f}")

    # Find best k
    best_k = max(accuracy_results, key=accuracy_results.get)
    best_accuracy = accuracy_results[best_k]

    print("\n" + "="*50)
    print("Best Model:")
    print(f"  k = {best_k}")
    print(f"  Test Accuracy = {best_accuracy:.6f}")
    print("="*50 + "\n")


def main():
    """Main pipeline execution."""
    print("Loading data...")
    train_df, test_df = load_data()
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    print("\nStage 1: Training Multiple Linear Regression for BloodPressure prediction...")
    test_df_modified, reg_model = stage1_regression(train_df, test_df)
    print("BloodPressure predictions completed. Test dataset updated.")

    print("\nStage 2: Training KNN classifiers (k=1 to 19) for Outcome prediction...")
    accuracy_results = stage2_knn_classification(train_df, test_df_modified)

    print_results(accuracy_results)


if __name__ == "__main__":
    main()
