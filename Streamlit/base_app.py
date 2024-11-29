"""
Simple Streamlit webserver application for serving developed classification
models.

Author: ExploreAI Academy.

Note:
---------------------------------------------------------------------
Please follow the instructions provided within the README.md file
located within this directory for guidance on how to use this script
correctly.
---------------------------------------------------------------------

Description: This file is used to launch a minimal streamlit web
application. You are expected to extend the functionality of this script
as part of your predict project.

For further help with the Streamlit framework, see:

https://docs.streamlit.io/en/latest/

"""

import streamlit as st
import joblib
import os
import pandas as pd
import re
import requests
from io import BytesIO  # Import BytesIO
import numpy as np
import requests
import base64
import pickle

# Function to fetch and load a pickle file from a URL
def load_pickle_from_url(url, token):
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Use BytesIO to treat the bytes response as a file-like object
        return joblib.load(BytesIO(response.content))
    else:
        st.error(f"Error loading the file from URL: {url} (Status code: {response.status_code})")
        return None

# The main function where we will build the actual app
def main():
    """Anime recommender system"""

    st.title("Anime recommender system")
    #st.subheader("Predicting credit customer profit scores")
    st.image("https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/image.png", use_column_width=True)

Let me know if you need any further assistance or customization!



    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("A collaborative and content-based recommender system for a collection of anime titles, capable of accurately predicting how a user will rate an anime title they have not yet viewed, based on their historical preferences")
        st.markdown("""
            - Education
            - Technology
            - Business
            - Entertainment 
            - Sports
        """)

    if selection == "Prediction":
        st.info("Choose between using a content filter or colloborative filer - describe them")
        
        # Creating a text box for user input
       option = st.selectbox( 'Select a filter:', ('Content Based Filtering', 'Collaborative Based Filtering') )



        if st.button("Predict"):
            variables = np.array([Salary,
                                  Internal_PD,
                                  External_PD,
                                  Loan_amount,
                                  Banking_with_bank,
                                  Internal_utilisation,
                                  External_utilisation,
                                  Spend_percentage])

            # Replace with your GitHub personal access token
            #token = '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'
            # Use the raw URL for the pickle file
            #model_url = 'https://raw.githubusercontent.com/VuyiswaK/Workplace_project/main/Streamlit/prediction_model.pkl'

            #model = load_pickle_from_url(model_url, token)

            
            
            # Display the selected option
            st.write('You selected:', option)
            
            # Load models
            def load_model(option):
                if option == 'Content Based Filtering':
                    # Load model 1
                    model_url = 'https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/content_model.pkl'
                else:
                    # Load model 2
                    model_url = 'https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/collab_model.pkl'
                
                # Replace with your GitHub personal access token
                token = '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'
      
                return load_pickle_from_url(model_url, token)
            
            # Load the selected model
            model = load_model(option)
            
            if model is not None:
                st.write("Model loaded successfully!")
            else:
                st.write("Failed to load the model.")
                        
            
            if model is not None:
                prediction = model.predict([variables])
                st.success(f"Text Category: {prediction}")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
