import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

def load_dataset(file_path):
    # Load dataset from a CSV file
    return pd.read_csv(file_path)


def preprocess_data(df):
    # Preprocess the data (placeholder)
    # Example: Handle missing values, encode categorical variables etc.
    return df.fillna(0)  # Example of filling missing values with 0


def train_model(X, y):
    # Train a Random Forest model
    model = RandomForestRegressor()
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2


def save_model(model, file_name):
    # Save the trained model to a file
    joblib.dump(model, file_name)


def main():
    # Path to dataset
    file_path = 'data/crop_yield.csv'
    # Load and preprocess the data
    df = load_dataset(file_path)
    df = preprocess_data(df)
    # Split data into features and target variable
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model
    model = train_model(X_train, y_train)
    # Evaluate the model
    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}')
    # Save the model
    save_model(model, 'crop_yield_model.pkl')


if __name__ == '__main__':
    main()