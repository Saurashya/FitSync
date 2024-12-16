import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path, file_name):
    df = pd.read_csv(file_path + file_name)  
    return df

class DataPreprocessing:
    def __init__(self, dataframe, categorical_columns, numerical_columns):
        self.data = dataframe
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
    
    def check_missing_values(self):
        print(self.data.isnull().sum())

    def handle_missing_values(self, strategy="mean"):
        if strategy == "mean":
            return self.data.fillna(self.data.mean())
        elif strategy == "median":
            return self.data.fillna(self.data.median())
        elif strategy == "mode":
            return self.data.fillna(self.data.mode().iloc[0])
        elif strategy == "imputer":
            # Imputing missing values (if any) with the mean for numerical columns
            imputer = SimpleImputer(strategy='mean')
            self.data[self.numerical_columns] = imputer.fit_transform(
                self.data[self.numerical_columns]
                )            
        else:
            raise ValueError("Unsupported strategy. Use 'mean', 'median', 'mode' or 'imputer'.")
            
    def encode_categorical_columns(self):
        # If categorical columns are provided, ensure they are encoded
        if self.categorical_columns:
            self.data = pd.get_dummies(self.data, columns=self.categorical_columns)

    def scale_numerical_features(self):
        # Scaling numerical columns
        scaler = StandardScaler()
        self.data[self.numerical_columns] = scaler.fit_transform(self.data[self.numerical_columns])

    def preprocess_data(self):
        self.check_missing_values()
        # self.handle_missing_values()
        self.encode_categorical_columns()
        self.scale_numerical_features()
        return self.data