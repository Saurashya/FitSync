import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from recommendation_model import RecommendationModel, ModelEvaluation, prepare_data

# Constants for the dashboard
TITLE_ICON = ":bar_chart:"  
HIDE_STREAMLIT_STYLE = """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

# Load the data with caching to improve performance
@st.cache_data
def get_data(file_path: str, file_name: str):
    """Loads data from a CSV file and returns a DataFrame."""
    df = pd.read_csv(file_path + file_name)
    return df

# Function to plot different evaluation metrics
def plot_evaluation_metrics(models, mae, rmse, mse, precision_at_5):
    """
    Displays bar charts for different metrics.

    Parameters:
    - models: List of model names.
    - mae: List of Mean Absolute Error values.
    - rmse: List of Root Mean Squared Error values.
    - mse: List of Mean Squared Error values.
    - precision_at_5: List of Precision at k values.
    """

    # Define positions for bars
    x = np.arange(len(models))
    width = 0.2 

    # Create a subplot 
    fig = go.Figure()

    # Adding MAE bars
    fig.add_trace(go.Bar(
        x=models,
        y=mae,
        name='MAE',
        marker=dict(color='blue')
    ))

    # Adding RMSE bars
    fig.add_trace(go.Bar(
        x=models,
        y=rmse,
        name='RMSE',
        marker=dict(color='green')
    ))

    # Adding MSE bars
    fig.add_trace(go.Bar(
        x=models,
        y=mse,
        name='MSE',
        marker=dict(color='orange')
    ))

    # Adding Precision@5 bars
    fig.add_trace(go.Bar(
        x=models,
        y=precision_at_5,
        name='Precision@5',
        marker=dict(color='purple')
    ))

    # Update layout
    fig.update_layout(
        barmode='group',
        title='Model Evaluation Metrics',
        xaxis_title='Models',
        yaxis_title='Values',
        plot_bgcolor="rgba(0,0,0,0)",
        template="plotly_white"
    )

    # Display the figure
    st.plotly_chart(fig)

# Function to evaluate the recommendation model and display graphs
def display_graph(data):
    """
    Train different models and display their performance metrics using bar charts.
    """
    if st.button("Display Graph"):
        # Create RecommendationModel instance
        recommendation_model = RecommendationModel(data)

        # Train and evaluate models
        methods = ['SVD', 'ALS', 'KNN']
        metrics = {'mae': [], 'rmse': [], 'mse': [], 'precision_at_5': []}

        for method in methods:
            model, testset = recommendation_model.train_model(method=method)
            model_eval = ModelEvaluation(model, testset, method)
            # Evaluate model
            mae, rmse, mse = model_eval.evaluate_model()
            precision = model_eval.precision_at_k(k=5)
            
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['mse'].append(mse)
            metrics['precision_at_5'].append(precision)

        # Plot all the metrics in one chart
        plot_evaluation_metrics(methods, metrics['mae'], metrics['rmse'], metrics['mse'], metrics['precision_at_5'])

# Streamlit Dashboard
def streamlit_dashboard(df):
    # Title and Introduction Section
    st.title("Recommendation Model Evaluation Dashboard")
    st.markdown("""        
        This dashboard provides key insights and evaluations for different recommendation models such as **SVD**, **ALS**, and **KNN**. 
        You can view metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Precision@5 for these models.

        ### How to Use:
        - **Choose a Model**: Select a recommendation model from the dropdown.
        - **View Metrics**: Select an option to view evaluation metrics for the selected model.
        - **Display Graphs**: Click the "Display Graph" button to see a comparative analysis of all models.
        
        Use this dashboard to understand the performance of different recommendation techniques and choose the best one for your needs.
    """)
    
    # User Inputs
    st.subheader("Model Selection and Evaluation")
    selected_method = st.selectbox("Select a Recommendation Model:", options=['SVD', 'ALS', 'KNN'])

    st.markdown(f"### Recommendation Model Training and Evaluation")
    st.write("Select the metrics you want to view and analyze for the selected recommendation model.")

    data = prepare_data(df)

    # Create RecommendationModel instance
    recommendation_model = RecommendationModel(data)

    # Selectbox for choosing the action
    selected_choice = st.selectbox("Choose Metrics to Display", 
                                    options=["Metrics for SVD", 
                                            "Metrics for ALS", 
                                            "Metrics for KNN"])
    
    # Button to trigger display of the selected option
    if st.button("Display Selected Option"):
        if selected_choice == f"Metrics for {selected_method}":

            # Train the selected model
            model, testset = recommendation_model.train_model(method=selected_method)

            model_eval = ModelEvaluation(model, testset, selected_method)

            # Evaluate model
            mae, rmse, mse = model_eval.evaluate_model()

            # Calculate precision at k
            precision = model_eval.precision_at_k(k=5)

            st.markdown(f"#### Evaluation Metrics for {selected_method}:")
            st.write(f'MAE: {mae:.4f}')
            st.write(f'RMSE: {rmse:.4f}')
            st.write(f'MSE: {mse:.4f}')
            st.write(f'Precision@5: {precision:.4f}')
    
    st.divider()

    st.markdown(f"### Graphical Representation")
    st.write("Click the button below to view a comparative analysis of the evaluation metrics for all the models.")

    display_graph(data)

# Main function to set up the dashboard
def app():
    # Set the dashboard title and header
    st.title(f"{TITLE_ICON} Recommendation Model Analysis")
    st.header("Analytic Dashboard")
    st.markdown("""
        This dashboard provides key insights into the performance of different recommendation models.
        You can view the evaluation metrics and compare them to choose the best model for your use case.
    """)
    st.divider()

    # File path and name for the data
    file_path = 'health_data/'
    file_name = 'exercise_dataset3000.csv'

    # Load and preprocess the data
    df = get_data(file_path, file_name)

    # Display the dashboard
    streamlit_dashboard(df)