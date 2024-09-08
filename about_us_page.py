import streamlit as st

def display_about_us_page():
    #st.title("About Us")
    col1, col2 = st.columns(2)

    # Display the logo
    with col1:
        st.image("assets/UFR.png", width=200)  # Adjust the width as needed

    with col2:
        st.image("assets/esprit.png", width=200)
    st.write("""
    ## About Project

    Welcome to our Stock Forecast App! This project is part of a research collaboration with Dr. Jingtao Yao at the University of Regina, Canada.
    ### Team
    Our team is composed of Our team consists of Nermine Ben Amara, a Data Science student at ESPRIT School of Engineering, and Dr. Jingtao Yao, Professor of Computer Science at the University of Regina, Canada.

    ### Our Vision
    To be the leading provider of stock forecast solutions that empower users to make smarter investment decisions.

    ### Contact Us
    We are always open to feedback and collaboration. Please reach out to us using the contact details below:
    
    - **Email**: nermine.benamara@esprit.tn
    - **Email**: jingtao.yao@uregina.ca
    - **Phone**: +216 51888230
    
    """)

    # Display your photo
    st.image("assets/nermine.jpg", caption="Nermine Ben Amara", width=150)  # Adjust the width and caption as needed

    st.write("""
    ## Follow Us
    - [LinkedIn "Nermine"](https://www.linkedin.com/in/nerminebenamara/)
    - [LinkedIn "Jingtao Yao"](https://www.linkedin.com/in/jingtao-yao-%E5%A7%9A%E9%9D%99%E6%B6%9B-%E3%80%80-88608b24/?originalSubdomain=ca)
    """)


