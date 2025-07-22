
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model():
    data_path = os.path.join("data", "salary_prediction_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna()
    df = df[['Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender', 'Salary']]

    X = df.drop('Salary', axis=1)
    y = df['Salary']

    categorical = ['Education', 'Location', 'Job_Title', 'Gender']
    numerical = ['Experience', 'Age']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', StandardScaler(), numerical)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
    joblib.dump((X_test, y_test), "test_data.pkl")
    print("✅ Model and test data saved.")

if __name__ == "__main__":
    train_and_save_model()
