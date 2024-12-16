import streamlit as st
from nav_menu import MultiApp
import home, basic_reco, collaborativefilter, modelevaluation

# Constants for styling and page configuration
PAGE_ICON = ":bar_chart:"  
LAYOUT = "wide"  



# Main function to run the app
def main():
    # Configuration of Streamlit page
    st.set_page_config(
        page_title="Personalized Exercise Recommendations", 
        page_icon=PAGE_ICON, 
        layout=LAYOUT,
        initial_sidebar_state="expanded"
    )

    # Sidebar information
    with st.sidebar:
        st.markdown("## Navigation Menu")
        st.markdown("""
        - **Home:** Overview and instructions.
        - **Basic Recommendation:** Basic Recommendation.
        - **Collaborative Filtering:** Personalized recommendations.
        - **Model Evaluation:** Model evaluations and visualizations.
        """)
    
    # Define the applications
    app = MultiApp()

    # Add the different pages to the MultiApp
    app.add_app("Home", home.app)
    app.add_app("Basic Recommendation", basic_reco.app)
    app.add_app("Collaborative Filtering", collaborativefilter.app)
    app.add_app("Model Evaluation", modelevaluation.app)



    # Run the main app
    if __name__ == '__main__':
        app.run()

if __name__ == "__main__":
    main()