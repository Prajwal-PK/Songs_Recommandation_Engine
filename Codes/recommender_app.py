import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
import os

# Define file paths
data_file = 'Codes/data.pkl'
model_file = 'Codes/random_forest_model.pkl'

# Debugging: Print file paths and check if files exist
st.write("Current Working Directory:", os.getcwd())
st.write("Checking if data file exists:", os.path.isfile(data_file))
st.write("Checking if model file exists:", os.path.isfile(model_file))

try:
    # Load data and model (ensure correct file paths)
    data = pickle.load(open(data_file, 'rb'))
    RF = pickle.load(open(model_file, 'rb'), encoding='latin1')
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
except pickle.PickleError as e:
    st.error(f"Error loading pickle file: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")

# Check if the model has feature names
try:
    model_features = RF.feature_names_in_
except AttributeError as e:
    st.error(f"Error accessing model features: {e}")

# Ensure that the data has the same feature columns
try:
    data_features = data[model_features]
except KeyError as e:
    st.error(f"Feature columns missing in data: {e}")
except Exception as e:
    st.error(f"Unexpected error processing data features: {e}")

# Prepare the list of songs
songs = data['name'].unique().tolist()  # Convert to list for st.selectbox compatibility

def recommend(user_song_name):
    if user_song_name in data['name'].values:  # Check if song exists
        user_song_index = data[data['name'] == user_song_name].index[0]
        user_song_features = data_features.iloc[user_song_index, :]  # Use consistent feature columns

        try:
            # Predict popularity for chosen song and other songs
            user_song_pred = RF.predict(user_song_features.values.reshape(1, -1))[0]
            song_preds = RF.predict(data_features)  # Use consistent feature columns

            # Recommend top N songs with similar predicted popularity
            N = 10
            diffs = np.abs(song_preds - user_song_pred)
            closest_indices = diffs.argsort()[:N]
            recommended_songs = data.loc[data.index[closest_indices], 'name'].tolist()
            return recommended_songs
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return "An error occurred during prediction."
    else:
        return "Sorry, that song is not in our database."

st.title('Song Recommendation Engine')

selected_song = st.selectbox('Choose a song you like:', songs)

if st.button('Recommend'):
    recommendations = recommend(selected_song)
    if isinstance(recommendations, list):  # Check if recommendations are available
        st.header("Recommendations for you:")
        for song in recommendations:
            st.write(song)
    else:
        st.write(recommendations)  # Display error message if song not found
