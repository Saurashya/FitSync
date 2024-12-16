# Personalized Exercise Recommendation System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-0.90%2B-orange.svg)

## Overview

Welcome to the Personalized Exercise Recommendation System! This dashboard leverages advanced collaborative filtering techniques and machine learning models to provide customized exercise recommendations based on user profiles. The goal is to help users find exercises that are best suited to their needs, improving their fitness journey in an informed and personalized manner.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [License](#license)

## Features

- **User-Based Collaborative Filtering:** Find users with similar exercise profiles and recommend exercises based on their preferences.
- **Model-Based Recommendations:** Utilize algorithms like SVD, ALS, and KNN to predict user preferences and provide recommendations.
- **Model Evaluation:** Evaluate recommendation models using metrics like MAE, RMSE, and Precision@k.
- **Interactive Visualizations:** Display evaluation metrics and insights through interactive charts and graphs.
- **User-Friendly Dashboard:** An intuitive Streamlit-based interface for easy navigation and interaction.

## Installation

### Prerequisites

- Python 3.8 or higher
- Anaconda (recommended for environment management)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Saurashya/fit-sync.git
   cd Exercise-Recommendation-System
   ```

2. **Create a virtual environment:**

   ```bash
   conda create --name exercise-reco
   conda activate exercise-reco
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**

   Place your dataset (e.g., `exercise_dataset.csv`) in the `health_data/` directory.

5. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

6. **Navigate to the application:**

   Open your web browser and go to `http://localhost:8501`.

## Usage

1. **Home Page:**
   - Overview of the application and how to navigate through it.
   - Learn about different features available in the dashboard.

2. **Basic Recommendation:**
   - Discover basic exercise recommendations.
   - Observe impact of various factors BMI, exercise intensity etc.

3. **Collaborative Filtering:**
   - Discover personalized exercise recommendations based on user profiles.
   - View similar users and their preferred exercises.

4. **Recommendation Model Evaluation:**
   - Evaluate different recommendation models.
   - Visualize metrics such as MAE, RMSE, and Precision@k using bar charts.
   - Choose a model and get insights into its performance.

5. **Input Requirements:**
   - Provide necessary user inputs such as User ID, Age, Gender, and BMI category for personalized recommendations.
   - Select models and view recommendations based on various algorithms.

## Project Structure

```
Exercise-Recommendation-System/
│
├── app.py                      # Main Streamlit application file
├── README.md                   # Project README file
├── requirements.txt            # Python dependencies
├── health_data/                # Directory for datasets
│   └── exercise_dataset.csv
├── collaborativefiltering.py   # Collaborative Filtering Logic
├── recommendation_model.py     # Recommendation Model Evaluation Logic
├── nav_menu.py                 # Streamlit Navigation Menu Logic
├── home.py                     # Streamlit Home Page 
├── basic_reco.py               # Streamlit Basic Recommendation Logic
├── collaborativefilter.py      # Streamlit Collaborative Filtering Logic
└── modelevaluation.py          # Streamlit Recommendation Model Evaluation Logic

```

## Technologies Used

- **Python:** Programming language used for implementing the logic.
- **Streamlit:** Framework used for creating interactive dashboards.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical operations.
- **Scikit-learn:** Machine learning models and evaluation.
- **Surprise:** Recommendation algorithms.
- **Plotly:** Interactive plots and visualizations.


