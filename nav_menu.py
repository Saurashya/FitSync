import streamlit as st
from streamlit_option_menu import option_menu


# Sidebar navigation
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        # Sidebar navigation
        with st.sidebar:
            selected_app = option_menu(
                menu_title="Navigation Menu",  
                options=["Home", "Basic Recommendation", "Collaborative Filtering", "Model Evaluation"],
                icons=["house-fill", "star", "bar-chart", "bar-chart"],  
                menu_icon="cast",
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "5!important", "background-color": '#004080'},  
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {"color": "white", "font-size": "18px", "text-align": "center"},
                    "nav-link-selected": {"background-color": "#0288d1"},  
                }
            )

        # Run the selected application
        for app in self.apps:
            if app['title'] == selected_app:
                app['function']()