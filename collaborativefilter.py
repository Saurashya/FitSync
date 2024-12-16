import pandas as pd
import streamlit as st
from collaborativefiltering import UserBasedCollaborativeFiltering
import math

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

# Streamlit Dashboard
def streamlit_dashboard(df):
    # Dashboard title and introduction
    st.title("Exercise Recommendation System")
    st.markdown("""
        Welcome to the **Exercise Recommendation System**! This dashboard leverages user-based collaborative filtering to provide personalized exercise recommendations.
        You can explore various insights based on user profiles, such as recommended exercises, similarity scores with other users, and fitness scores based on selected features.
        """)
    
    # Input user profile
    st.subheader("User Input:")
    st.markdown("""
        Please provide the necessary inputs below to get personalized recommendations and insights. You can explore:
        - **Similarity Score**: Find users similar to you based on your user ID.
        - **Recommended Exercises**: Discover exercises recommended for you based on your profile.
        - **User Fitness Score**: Get a fitness score prediction for a specific feature based on similar users.
        - **User Profile Summary**: View a summary of your profile details.
    """)

    st.divider()

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_user_id = st.number_input("User ID", min_value=1, format="%d")
    with col2:
        selected_number_users = st.number_input("Number of Users", min_value=1, format="%d")
    with col3:
        feature_column = ['Calories Burn', 'Dream Weight', 'Duration', 'Heart Rate', 'BMI', 'Exercise Intensity']
        selected_feature_column = st.selectbox("Feature:", options=feature_column)

    st.divider()

    # Initialize the User-Based Collaborative Filtering class
    ucf = UserBasedCollaborativeFiltering(df)

    # Compute cosine similarity between users
    similarity_matrix = ucf.compute_similarity()

    # Instructions for using the dashboard
    st.markdown(f"### User-Based Collaborative Filtering")

    st.markdown("#### Instructions:")
    st.write("""
        - **Similarity Score**: Enter a valid User ID and the number of similar users you want to see.
        - **Recommended Exercises**: Enter a valid User ID to view recommended exercises.
        - **User Fitness Score**: Enter a valid User ID, select a feature, and enter the number of similar users for fitness score prediction.
        - **User Profile Summary**: Enter a valid User ID to view your profile details.
    """)

    # Selectbox for choosing the action
    selected_choice = st.selectbox("Select an Option", 
                                   options=["Similarity Score", 
                                            "Recommended Exercises", 
                                            "User Fitness Score", 
                                            "User Profile Summary"])
    
    # Button to trigger display of the selected option
    if st.button("Display Selected Option"):
        try:
            if selected_user_id <= 0 or selected_number_users <= 0:
                st.warning("Please enter valid values for User ID and Number of Users.")
                return

            if selected_choice == "Similarity Score":
                # Get the top similar users for the user ID
                similar_users = ucf.get_top_n_similar_users(user_id=selected_user_id, n=selected_number_users)
                st.markdown(f"#### Top {selected_number_users} similar users for User {selected_user_id}:")
                for user, score in similar_users:
                    st.markdown(f"- **User {user}**, Similarity Score: {score:.3f}")
            
            elif selected_choice == "Recommended Exercises":
                # Recommend exercises for a user
                recommended_exercises = ucf.recommend_exercises(user_id=selected_user_id)    
                st.markdown(f"#### Recommended Exercises for User {selected_user_id}:")
                for exercise, calories, sets in recommended_exercises:
                    st.markdown(f"- **{exercise}**: {math.floor(calories)} calories burned, Sets: {sets}")
            
            elif selected_choice == "User Fitness Score":
                # Predict a fitness score for a user
                predicted_ei = ucf.predict_fitness_score(user_id=selected_user_id, feature_column=selected_feature_column, top_n=selected_number_users)
                st.markdown(f"#### Predicted {selected_feature_column} for User {selected_user_id} based on similar users: **{predicted_ei:.2f}**")
            
            elif selected_choice == "User Profile Summary":
                # Get user profile summary
                profile_summary = ucf.get_user_profile_summary(user_id=selected_user_id)
                if isinstance(profile_summary, dict):
                    st.markdown(f"#### Profile Summary for User {selected_user_id}:")
                    for description, value in profile_summary.items():
                        st.markdown(f"- **{description}:** {value}")
                else:
                    st.warning(profile_summary)
        except KeyError:
            st.error(f"No data found for User ID {selected_user_id}.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Main function to set up the dashboard
def app():
    st.title(f"{TITLE_ICON} Exercise Recommendation Dashboard")

    st.header("Collaborative Filtering for Exercise Recommendations")
    st.markdown("""
        This dashboard provides personalized exercise recommendations based on user profiles using collaborative filtering.
        You can explore recommendations based on user similarity, exercise intensity, and more.
        Use the input options to customize your experience and discover exercises that best match your fitness goals!
    """)
    st.divider()

    file_path = 'health_data/'
    file_name = 'exercise_dataset3000.csv'

    # Load and preprocess the data
    df = get_data(file_path, file_name)

    # Display the dashboard
    streamlit_dashboard(df)
