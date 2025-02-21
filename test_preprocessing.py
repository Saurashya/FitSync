import pytest
import pandas as pd
from preprocessing import load_data, DataPreprocessing

# Load the dataset
file_path = "D:/Coding/FitSync/health_data/"
file_name = "exercise_dataset3000.csv"
df = load_data(file_path, file_name)

def test_load_data():
    print("Running test_load_data")
    # Check if the data is loaded correctly
    assert df.shape[0] > 0  # Ensure there are rows
    assert df.shape[1] == 12  # Ensure there are 12 columns
    print(df.head())  # Print the first few rows of the DataFrame

def test_check_missing_values():
    print("Running test_check_missing_values")
    dp = DataPreprocessing(df, categorical_columns=['Exercise', 'Gender', 'Weather Conditions'], numerical_columns=['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity'])
    
    dp.check_missing_values()
    print(dp.data)  # Print the DataFrame
    assert dp.data.isnull().sum().sum() >= 0  # Check for missing values

def test_handle_missing_values():
    print("Running test_handle_missing_values")
    dp = DataPreprocessing(df, categorical_columns=['Exercise', 'Gender', 'Weather Conditions'], numerical_columns=['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity'])
    
    df_filled = dp.handle_missing_values(strategy="mean")
    print(df_filled)  # Print the DataFrame
    assert df_filled.isnull().sum().sum() == 0  # Ensure no missing values

def test_encode_categorical_columns():
    print("Running test_encode_categorical_columns")
    dp = DataPreprocessing(df, categorical_columns=['Exercise', 'Gender', 'Weather Conditions'], numerical_columns=['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity'])
    
    dp.encode_categorical_columns()
    print(dp.data)  # Print the DataFrame
    assert 'Exercise_Squats' in dp.data.columns  # Check for one-hot encoded columns
    assert 'Gender_Male' in dp.data.columns
    assert 'Weather Conditions_Rainy' in dp.data.columns

def test_scale_numerical_features():
    print("Running test_scale_numerical_features")
    dp = DataPreprocessing(df, categorical_columns=['Exercise', 'Gender', 'Weather Conditions'], numerical_columns=['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity'])
    
    dp.scale_numerical_features()
    print(dp.data)  # Print the DataFrame
    assert dp.data['Calories Burn'].mean() == pytest.approx(0, abs=1e-6)
    assert dp.data['Dream Weight'].mean() == pytest.approx(0, abs=1e-6)

def test_preprocess_data():
    print("Running test_preprocess_data")
    dp = DataPreprocessing(df, categorical_columns=['Exercise', 'Gender', 'Weather Conditions'], numerical_columns=['Calories Burn', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity'])
    
    processed_df = dp.preprocess_data()
    print(processed_df)  # Print the DataFrame
    assert 'Exercise_Squats' in processed_df.columns
    assert 'Gender_Male' in processed_df.columns
    assert 'Weather Conditions_Rainy' in processed_df.columns
    assert processed_df['Calories Burn'].mean() == pytest.approx(0, abs=1e-6)