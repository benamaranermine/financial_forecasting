import streamlit as st

# Import the page functions
from homepage import display_homepage
from demo_page import display_demo_page
from about_us_page import display_about_us_page

# Define the main function to handle page routing
def main():
    # Custom CSS for the sidebar
    st.markdown("""
        <style>
        /* Sidebar background color */
        .st-emotion-cache-6qob1r {
            background-color: #000000;  /* Black background */
            color: #ffffff;  /* White text color */
        }
        
        /* Sidebar text color */
        .st-emotion-cache-6qob1r * {
            color: #ffffff;  /* Ensure all text is white */
        }

        /* Sidebar radio buttons color */
        .st-emotion-cache-6qob1r .stRadio {
            background-color: #000000;  /* Black background for radio buttons */
        }
        
        .st-emotion-cache-6qob1r .stRadio label {
            color: #ffffff;  /* White text for radio button labels */
        }

        /* Sidebar title color */
        .st-emotion-cache-6qob1r h1 {
            color: #ffffff;  /* White color for title text */
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    page = st.sidebar.radio("Go to", ["Home", "Demo", "About Us"])

    # Display the selected page
    if page == "Home":
        display_homepage()
    elif page == "Demo":
        display_demo_page()
    elif page == "About Us":
        display_about_us_page()

if __name__ == "__main__":
    main()
