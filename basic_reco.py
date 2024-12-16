import pandas as pd
import streamlit as st

# Constants for the dashboard
TITLE_ICON = ":star:"
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

# Function to categorize BMI
def categorize_bmi(df):
    """
    Categorizes individuals into different BMI groups.

    Parameters:
    - df: DataFrame containing BMI values.

    Returns:
    - DataFrame with an additional column 'BMI Category' indicating the BMI group.
    """
    categories = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    df['BMI Category'] = pd.cut(df['BMI'], 
                                bins=[-float('inf'), 18.5, 24.9, 29.9, float('inf')], 
                                labels=categories)
    return df

# Function to recommend exercises based on gender, age, and BMI category
def get_recommended_exercises(df, gender, age, bmi_category):
    """
    Returns recommended exercises based on gender, age, and BMI category.

    Parameters:
    - df: DataFrame containing exercise data.
    - gender: The gender of the user (e.g., 'Male' or 'Female').
    - age: The age of the user.
    - bmi_category: The BMI category of the user (e.g., 'Underweight', 'Normal weight', 'Overweight', 'Obesity').

    Returns:
    - recommended_exercises: A list of recommended exercises based on the user's profile.
    """
    # Filter the dataframe based on gender
    filtered_df = df[df['Gender'] == gender]

    # Further filter based on age groups
    if age < 18:
        filtered_df = filtered_df[filtered_df['Age'] < 18]
    elif 18 <= age < 35:
        filtered_df = filtered_df[(filtered_df['Age'] >= 18) & (filtered_df['Age'] < 35)]
    elif 35 <= age < 50:
        filtered_df = filtered_df[(filtered_df['Age'] >= 35) & (filtered_df['Age'] < 50)]
    elif 50 <= age <= 60:
        filtered_df = filtered_df[(filtered_df['Age'] >= 50) & (filtered_df['Age'] <= 60)]
    else:
        return "Invalid age input."

    # Further filter based on BMI category
    filtered_df = filtered_df[filtered_df['BMI Category'] == bmi_category]

    # Check if there are any matching records
    if filtered_df.empty:
        return "No matching data found for the given profile."

    # Get the list of unique recommended exercises
    recommended_exercises = filtered_df['Exercise'].unique().tolist()

    # Check if any exercises were found
    if not recommended_exercises:
        return "No recommended exercises found for the selected profile."

    return recommended_exercises

# Function to categorize exercise intensity
def categorize_exercise_intensity(intensity):
    """
    Categorizes exercise intensity into different groups.
    - Low: 1 to 3
    - Moderate: 4 to 6
    - High: 7 to 10
    """
    if 1 <= intensity <= 3:
        return 'Low'
    elif 4 <= intensity <= 6:
        return 'Moderate'
    elif 7 <= intensity <= 10:
        return 'High'
    else:
        return 'Unknown'  

# Apply the exercise intensity categorization to the dataframe
def apply_exercise_intensity(df):
    df['Intensity Category'] = df['Exercise Intensity'].apply(categorize_exercise_intensity)
    return df

# Function to recommend exercises based on intensity and duration
def get_recommended_exercises_intensity(df, intensity_category, duration):
    """
    Returns recommended exercises based on intensity category and duration.

    Parameters:
    - df: DataFrame containing exercise data.
    - intensity_category: The category of exercise intensity ('Low', 'Moderate', 'High').
    - duration: The maximum duration for the recommended exercises.

    Returns:
    - recommended_exercises: A list of recommended exercises based on intensity and duration.
    """
    # Filter the dataframe
    filtered_df = df[df['Intensity Category'] == intensity_category]
    filtered_df = filtered_df[filtered_df['Duration'] <= duration]

    # Check if there are any matching records
    if filtered_df.empty:
        return "No matching data found for the given profile."

    # Get the list of unique recommended exercises
    recommended_exercises = filtered_df['Exercise'].unique().tolist()

    # Check if any exercises were found
    if not recommended_exercises:
        return "No recommended exercises found for the selected profile."

    return recommended_exercises

# Streamlit Dashboard
def streamlit_dashboard(df):
    st.title("Basic Exercise Recommendations")

    # Input user profile
    st.subheader("User Input:")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = df['Gender'].unique().tolist()
        selected_gender = st.selectbox("Gender:", options=gender) 
    with col2:
        min_age = df['Age'].min()
        max_age = df['Age'].max()
        age = st.slider("Age", min_value=min_age, max_value=max_age, value=(min_age + max_age) // 2)
    with col3:
        bmi_category = df['BMI Category'].unique().tolist()
        selected_bmi_category = st.selectbox("BMI Category:", options=bmi_category)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        intensity_category = df['Intensity Category'].unique().tolist()
        selected_intensity_category = st.selectbox('Intensity Category:', options=intensity_category)  
    with col2:
        min_duration = df['Duration'].min()
        max_duration = df['Duration'].max()
        duration = st.slider("Duration", min_value=min_duration, max_value=max_duration, value=(min_duration + max_duration) // 3)

    st.divider()

    st.markdown("### Exercise Recommendation Overview")
    st.markdown("""
    This dashboard provides basic exercise recommendations based on user inputs.
    You can explore recommendations based on different criteria such as:
    - **Gender, Age and BMI**: Get recommendations tailored to your gender, age group and BMI category.
    - **Intensity and Duration**: Find exercises that match your preferred intensity level and workout duration.
    """)
    st.markdown("#### Instructions:")
    st.write("- For recommendations by Age and BMI, provide your gender, age, and BMI category.")
    st.write("- For recommendations by Intensity, provide your preferred intensity category and workout duration.")

    # Selectbox for choosing the action
    selected_choice = st.selectbox("Choose Recommendation Type", 
                                    options=["Recommended Exercises by Age and BMI", 
                                            "Recommended Exercises by Intensity"])
    
    # Button to trigger display of the selected option
    if st.button("Display Recommendation"):
        if selected_choice == "Recommended Exercises by Age and BMI":
            recommended_exercises = get_recommended_exercises(df, selected_gender, age, selected_bmi_category)
            if isinstance(recommended_exercises, list):
                st.write(f"Recommended Exercises for {selected_gender}, Age {age}, BMI Category {selected_bmi_category}:")
                for exercise in recommended_exercises:
                    st.write(f"- {exercise}")
            else:
                st.write(recommended_exercises)

        elif selected_choice == "Recommended Exercises by Intensity":
            recommended_exercises = get_recommended_exercises_intensity(df, selected_intensity_category, duration)
            if isinstance(recommended_exercises, list):
                st.write(f"Recommended Exercises for {selected_intensity_category} Intensity Category with <= {duration} min duration:")
                for exercise in recommended_exercises:
                    st.write(f"- {exercise}")
            else:
                st.write(recommended_exercises)

# Main function to set up the dashboard
def app():
    st.title(f"\U0001F3CB\uFE0F Basic Recommendation")
    st.header("Exercise Recommendation Dashboard")
    st.markdown("""
    This dashboard provides key insights into exercise recommendations based on user profiles.
    You can filter recommendations based on your personal characteristics, such as age, gender, BMI, and exercise intensity.
    Explore and discover exercises that best match your fitness goals!
    """)
    st.divider()

    file_path = 'health_data/'
    file_name = 'exercise_dataset3000.csv'

    # Load and preprocess the data
    df = get_data(file_path, file_name)
    df = categorize_bmi(df)
    df = apply_exercise_intensity(df)

    # Display the dashboard
    streamlit_dashboard(df)

if __name__ == "__main__":
    app()
