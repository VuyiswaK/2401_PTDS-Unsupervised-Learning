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
import numpy as np
import operator # <-- Convenient item retrieval during iteration 
import heapq 

##
def content_generate_rating_estimate(title, user, rating_data, k=10, threshold=0.0):
    # Convert the book title to a numeric index for our 
    # similarity matrix
    #print(title)
    b_idx = indices[title]
    #print(b_idx)
    neighbors = [] # <-- Stores our collection of similarity values 
     
    # Gather the similarity ratings between each book the user has rated
    # and the reference book 
    for index, row in rating_data[rating_data['user_id']==user].iterrows():
        #print(row['name'])
        #print(b_idx-1, indices[row['name']]-1)
        sim = cosine_sim_Tags[b_idx-1, indices[row['name']]-1]
        neighbors.append((sim, row['rating']))
    # Select the top-N values from our collection
    k_neighbors = heapq.nlargest(k, neighbors, key=lambda t: t[0])

    # Compute the weighted average using similarity scores and 
    # user item ratings. 
    simTotal, weightedSum = 0, 0
    for (simScore, rating) in k_neighbors:
        # Ensure that similarity ratings are above a given threshold
        if (simScore > threshold):
            simTotal += simScore
            weightedSum += simScore * rating
    try:
        predictedRating = weightedSum / simTotal
    except ZeroDivisionError:
        # Cold-start problem - No ratings given by user. 
        # We use the average rating for the reference item as a proxy in this case 
        predictedRating = np.mean(rating_data[rating_data['name']== title]['rating'])
    return predictedRating
    
##collab generate rating estimate

def collab_generate_rating_estimate(book_title, user, k=20, threshold=0.0):
    # Gather the k users most similar to the reference user
    #print('Gather the k users most similar to the reference user')
    
    # Sort the similarity values once and fetch the k most similar users and their similarity values
    sorted_sim_users = user_sim_df[user].sort_values(ascending=False).iloc[1:k+1]
    
    # Get the user IDs and their corresponding similarity values
    sim_users = sorted_sim_users.index
    user_values = sorted_sim_users.values
    
    # Extract ratings for the book title (avoiding repeated lookups)
    ratings = util_matrix.loc[sim_users, book_title].values
    
    # Filter out invalid ratings (NaN or below threshold) and compute weighted sum
    #print(f'Create weighted sum for each of the {len(sim_users)} users who have rated the item.')
    
    # Mask to filter out users with NaN ratings or similarity below threshold
    valid_mask = ~np.isnan(ratings) & (user_values >= threshold)
    
    # Apply mask to ratings and similarities
    valid_ratings = ratings[valid_mask]
    valid_similarities = user_values[valid_mask]
    
    # Calculate weighted sum
    if valid_ratings.size > 0:  # Ensure there are valid ratings
        weighted_sum = np.sum(valid_ratings * valid_similarities)
        total_weight = np.sum(valid_similarities)
        
        # Return the predicted rating as weighted average
        predicted_rating = weighted_sum / total_weight
    else:
        # If no valid ratings, return the average rating for the book
        print('No valid ratings found, returning average rating for the item.')
        predicted_rating = np.mean(util_matrix[book_title].dropna())
    
    return predicted_rating

# Function to fetch and load a pickle file from a URL


def load_pickle_from_url(url):
    response = requests.get(url) 
    if response.status_code == 200: 
        # Use BytesIO to treat the bytes response as a file-like object 
        return joblib.load(BytesIO(response.content)) 
    else:
        st.error(f"Error loading the file from URL: {url} (Status code: {response.status_code})") 
        return None
        
token = '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'
indices = load_pickle_from_url('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/indices.pkl')
util_matrix = load_pickle_from_url('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/util_matrix.pkl')
user_sim_df = load_pickle_from_url('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/user_sim_df.pkl')
sample = pd.read_csv('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/sample.csv')
data = pd.read_csv('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/data.csv')
cosine_sim_Tags = np.array(load_pickle_from_url('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/Streamlit/cosine_sim_Tags.pkl'))

# The main function where we will build the actual app
def main():
    """Anime recommender system"""

    st.title("Anime recommender system")
    #st.subheader("Predicting credit customer profit scores")
    st.image('https://raw.githubusercontent.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/main/image.png', use_column_width=True)

    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    if selection == "Information":
        st.info("A collaborative and content-based recommender system for a collection of anime titles, capable of accurately predicting how a user will rate an anime title they have not yet viewed, based on their historical preferences")
        st.markdown("""
            - Content Based Filtering: In Content-based filtering, we seek to make recommendations based on **how similar an item's properties or features are to those of other items.**
            - Collaborative Based Filtering: In collaborative-based filtering, **we measure the similarity between users to make recommendations**

            Performance metrics:

        """)

        
        perf = { 'Filter': ['Content Based', 'Collab Based'], 'RMSE': [0.69, 1.41], 'Time (sec)': [0.08, 2.05] } 
        df = pd.DataFrame(perf) 
        df.set_index('Filter', inplace=True)
        # Display the DataFrame as a table
        st.table(df)
        
        st.markdown("""
         By far content based filter is the best in terms of execution time and accuracy.
        """)


    
    if selection == "Prediction":
        #st.image("https://github.com/VuyiswaK/2401_PTDS-Unsupervised-Learning/blob/main/Streamlit/image.png", use_column_width=True)

        #st.info("Choose between using a content filter or colloborative filer - describe them and approximately how long it'll take to run")
        
        # Creating a text box for user input
        option = st.radio( 'Select a filter:', ('Content Based Filtering', 'Collaborative Based Filtering') )
        # Display the selected option
        st.write('You selected:', option)

        user = st.selectbox('User:', sample.user_id)
        name = st.selectbox('Anime name:', data.name)

            # Replace with your GitHub personal access token
            #token = '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'
            # Use the raw URL for the pickle file
            #model_url = 'https://raw.githubusercontent.com/VuyiswaK/Workplace_project/main/Streamlit/prediction_model.pkl'

            #model = load_pickle_from_url(model_url, token)
            
        # Load models
        def load_model(option):
            if option == 'Content Based Filtering':
                # Load model 1
                output = content_generate_rating_estimate(title=name, user=user, rating_data=data)
            else:
                # Load model 2
                output = collab_generate_rating_estimate(name,user)
            
            # Replace with your GitHub personal access token
            #token = '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'
  
            return output #load_pickle_from_url(model_url, token)

        if st.button("Predict"):
            prediction = load_model(option)
            st.success(f"User {user} will give the anime  {name} a rating of {np.round(prediction,2)}")



# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
