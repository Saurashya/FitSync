import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, KNNBasic, Dataset, Reader, BaselineOnly
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import accuracy

def prepare_data(df):
    reader = Reader(rating_scale=(1, 5))  
    df['Overall Fitness Score'] = 1 * df['Exercise Intensity'] + 0 * df['Calories Burn'] + 0 * df['Duration'] + 0 * df['Dream Weight']
    data = Dataset.load_from_df(df[['ID', 'Exercise', 'Overall Fitness Score']], reader)
    return data

class RecommendationModel:
    def __init__(self, data):
        self.data = data
    
    def train_model(self, method='SVD'):
        trainset, testset = train_test_split(self.data, test_size=0.25)
        
        if method == 'SVD':
            print("Training model using Singular Value Decomposition (SVD)...")
            model = SVD(n_factors= 20, n_epochs=10, lr_all=0.001, reg_all=0.01)
        elif method == "ALS":
            print("Training model using Alternating Least Squares (ALS)...")
            bsl_options = {"method": "als", "n_epochs": 10, "reg_u": 12, "reg_i": 5}
            model = BaselineOnly(bsl_options=bsl_options)
        elif method == 'KNN':
            print(f"Training model using K-Nearest Neighbors (KNN)...")
            param={'k': 10, 'min_k': 1, 'sim_options': {'name': 'cosine', 'user_based': True}}
            model = KNNBasic(param)
        else:
            raise ValueError("Invalid method. Choose 'SVD' or 'ALS' or 'KNN'")
        
        model.fit(trainset)
        return model, testset

class ModelEvaluation:
    def __init__(self, model, testset, modelname=None):
        self.model = model
        self.testset = testset
        self.modelname=modelname
        self.predictions = None

    def prediction(self):
        self.predictions = self.model.test(self.testset)

    def evaluate_model(self):
        if self.predictions is None:
            self.prediction()  
            print("Generated predictions...")

        print(f"Metrics for {self.modelname}")

        mae=accuracy.mae(self.predictions)
        rmse=accuracy.rmse(self.predictions)
        mse=accuracy.mse(self.predictions)

        return mae, rmse, mse

    def precision_at_k(self, k=5):
        if self.predictions is None:
            self.prediction()  
            print("Generated predictions...")

        # Calculate precision@k
        correct = 0
        total = 0
        
        for pred in self.predictions:
            if pred.est >= k:  # Predicted rating >= k
                if pred.r_ui >= k:  # True rating >= k
                    correct += 1
                total += 1
        
        precision = correct / total if total else 0
        return precision


def optimize_hyperparameters(data, algorithm='SVD'):

    if algorithm == 'SVD':

        # Define the parameter grid for SVD
        param_grid = {
            'n_factors': [20, 30, 50, 100],
            'n_epochs': [10, 20, 30, 50],
            'lr_all': [0.001, 0.002, 0.005, 0.01],
            'reg_all': [0.01, 0.02, 0.1, 0.15]
        }

        # Perform grid search
        grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
        grid_search.fit(data)
    
    elif algorithm == 'KNN':

        # Define the param_grid for KNN
        param_grid = {
            'k': [10, 20, 30],
            'min_k': [1, 5],
            'sim_options': {
                'name': ['cosine', 'pearson'],
                'user_based': [True, False]
            }
        }

        # Apply GridSearchCV to find best KNN parameters
        grid_search = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=5)
        grid_search.fit(data)
    
    else:
        print("Invalid algorithm..")

    # Get best parameters
    print(f'Best RMSE: {grid_search.best_score["rmse"]}')
    print(f'Best MAE: {grid_search.best_score["mae"]}')
    print(f'Best parameters: {grid_search.best_params["rmse"]}')



def plot_metrics(models, mae, rmse, mse, precision_at_5):

    # Define positions for bars
    x = np.arange(len(models))

    # Define the width of the bars
    width = 0.4

    # Create a figure with subplots for each metric
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))

    # Plot MAE
    ax[0].bar(x, mae, width, color='b')
    ax[0].set_title('MAE')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(models)
    ax[0].set_ylabel('MAE')

    # Plot RMSE
    ax[1].bar(x, rmse, width, color='g')
    ax[1].set_title('RMSE')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(models)
    ax[1].set_ylabel('RMSE')

    # Plot MSE
    ax[2].bar(x, mse, width, color='r')
    ax[2].set_title('MSE')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(models)
    ax[2].set_ylabel('MSE')

    # Plot Precision@5
    ax[3].bar(x, precision_at_5, width, color='y')
    ax[3].set_title('Precision@5')
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(models)
    ax[3].set_ylabel('Precision@5')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()