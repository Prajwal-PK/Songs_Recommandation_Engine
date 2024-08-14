import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

songs = pd.read_csv("https://raw.githubusercontent.com/Prajwal-PK/Songs_Recommandation_Engine/main/Datasets/cleandata.csv")
songs = songs.drop(columns = ['id','release_date'] )
import ast
def fetch_list(strings):
    l=[]
    for i in ast.literal_eval(strings):
        l.append(i)
        return l

songs['artists'] = songs['artists'].apply(fetch_list)
songs['artists'] = songs['artists'].apply(lambda x: " ".join(x))
songs['artists'] = songs['artists'].apply(lambda x: x.lower())

songs['decade'] = songs['year'].apply(lambda year : f'{(year//10)*10}s' )
songs = songs[songs['year']>2000]
songs['mode'] = songs['mode'].astype(bool)
songs['explicit'] = songs['explicit'].astype(bool)
import pandas as pd
from sklearn.model_selection import train_test_split
# Data setup
features = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo']
data = songs # Load your actual data here
X = data[features]
y = data['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
songs.head()


user_song_name = 'Dynamite'
song_index = data[data['name'] == user_song_name].index

if len(song_index) == 0:
    print(f"Sorry, we couldn't find '{user_song_name}' in our database.")
else:
    user_song_features = data.loc[song_index, features].values



    # Create and train the Random Forest model
    RF = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
    RF.fit(X_train, y_train)


# Get the feature names used during model training
model_features = RF.feature_names_in_

# Ensure that the data has the same feature columns
data_features = data[model_features]

songs = data['name'].unique().tolist()  # Convert to list for st.selectbox compatibility

def recommend(user_song_name):
    if user_song_name in data['name'].values:  # Check if song exists
        user_song_index = data[data['name'] == user_song_name].index[0]
        user_song_features = data_features.iloc[user_song_index, :]  # Use consistent feature columns

        # Predict popularity for chosen song and other songs
        user_song_pred = RF.predict(user_song_features.values.reshape(1, -1))[0]
        song_preds = RF.predict(data_features)  # Use consistent feature columns

        # Recommend top N songs with similar predicted popularity
        N = 10
        diffs = np.abs(song_preds - user_song_pred)
        closest_indices = diffs.argsort()[:N]
        recommended_songs = data.loc[data.index[closest_indices], 'name'].tolist()
        return recommended_songs
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

