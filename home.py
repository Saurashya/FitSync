import streamlit as st

# Constants for the dashboard
TITLE_ICON = ":house:"
HIDE_STREAMLIT_STYLE = """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

# Introduction section 
def display_introduction():
    st.title(":dart: Exercise Recommendation Dashboard")
    st.markdown("""
    ## Welcome to the Personalized Exercise Recommendation System!

    This dashboard is designed to provide you with customized exercise recommendations based on your personal profile. It utilizes advanced collaborative filtering techniques and machine learning models to help you find the best exercises suited to your needs.

    ### Explore the Dashboard:
    - **Home:** Learn about the application and get an overview of its functionalities.                
    - **Basic Recommendation:** Get basic recommendation module.
    - **Collaborative Filtering:** Discover recommendations using user-based and item-based collaborative filtering. Get insights on similar users and preferred exercises.
    - **Model Evaluation:** Dive into model-based recommendations. Evaluate different models like SVD, ALS, and KNN, and visualize their performance metrics.

    ### How to Use This Dashboard:
    1. **Select a Section:** Use the sidebar menu to navigate between different sections.
    2. **Provide Your Input:** Depending on the selected section, provide your profile data like User ID, Age, Gender, BMI, etc.
    3. **View Recommendations:** Get personalized exercise recommendations, fitness scores, or model evaluations.

    **Let's get started!** Use the sidebar to navigate through the options.
    """)



# Main function to set up the dashboard
def app():
    st.title(f"{TITLE_ICON} Home")

    # Display introduction on the Home page
    display_introduction()

    st.divider()