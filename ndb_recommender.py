import os
import json
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import scipy.sparse as sp

from utilities import utils
from model import model_knn

import streamlit as st
from streamlit import caching
import streamlit.components.v1 as components

# Read Data
settings_root = './settings/'
ndb_tracks_df = pd.read_pickle('./data/ndb_mpd_tracks.compress', compression='gzip')
tracks_df = pd.read_pickle('./data/tracks_ndb.compress', compression='gzip')
track_title = tracks_df['track_name'].map(str) + " by " + tracks_df['artist_name']
tracks_to_tidx = pd.Series(track_title.values, index=tracks_df.tid).to_dict()
tracks_to_spidx = pd.Series(tracks_df.spid, index=tracks_df.tid).to_dict()
sparse_matrix = sp.load_npz('./data/matrix_playlistTrackRating_ndb.npz')

def SaveSpotifyPlaylist(playlistsTracksSPIDs):

    # Reading Spotify web API credentials from settings.env hidden file
    settingsfile = os.path.join(settings_root, 'settings.env')
    with open(settingsfile) as f:
        env_vars = json.loads(f.read())
    
    # Authorization flow
    client_id = env_vars['SPOTIPY_CLIENT_ID']
    client_secret = env_vars['SPOTIPY_CLIENT_SECRET']
    redirect_uri = env_vars['SPOTIPY_REDIRECT_URI']
    username = env_vars['SPOTIPY_USER']
    playlistID = env_vars['PLAYLIST_ID']
    scope = 'playlist-read-private playlist-modify-public'

    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)

    if token:
        sp = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)  
    
    # Save Random Playlist Tracks
    results = sp.user_playlist_replace_tracks(username, playlistID, playlistsTracksSPIDs)
    
    if results:
        return results['snapshot_id']
    else:
        return 'ERROR'

def main():

    # Sidebar Filter
    st.sidebar.image('./media/logo_NDB.png', use_column_width=True)
    values = ['<All>', 2016, 2017, 2018, 2019, 2021]
    edition = st.sidebar.selectbox('Select NDB Edition:', values,)
    topk = st.sidebar.slider('Number of Recommended Tracks:', 0, 50, 20, 10)
    generate_playlist = st.sidebar.button('Recommend...')

    if generate_playlist:

        if edition=='<All>':
            edition = 0
        playlist = utils.GenerateRandomNDBPlaylist(ndb_tracks_df, edition)
        if edition==0:
            edition = '<All>'
        input_playlist = [tracks_to_tidx[x] for x in playlist]
        df_edition = pd.DataFrame(input_playlist,  
                                columns=['title'],
                                index=pd.RangeIndex(start=1, stop=len(input_playlist)+1))
        df_edition = df_edition.style.set_properties(**{'text-align': 'left'})
        st.sidebar.text('Playlist Tracks from NDB Edition: {}'.format(edition))
        st.sidebar.write(df_edition)  

        # Recommends k Nearest Neighbors
        kNN_model = model_knn.nneighbors_model('knn', 25, 'cosine', sparse_matrix)
        sp_playlist = utils.BuildSparsePlaylist(playlist, sparse_matrix.shape[1])
        recommends_knn = kNN_model.predict_random(sp_playlist, topk, 25)
        recommends_knn_rank = [tracks_to_tidx[x] for x in recommends_knn]
        df_recommends = pd.DataFrame(recommends_knn_rank,
                                    columns=['title'],
                                    index=pd.RangeIndex(start=1, stop=len(recommends_knn)+1, name='rank'))
        df_recommends = df_recommends.style.set_properties(**{'text-align': 'left'})    

        st.title('Recommended Tracks for Edition: {}'.format(edition))
        

        # Map Tracks IDs to SPIDs
        recommendedTracks_SPIDs = [tracks_to_spidx[x] for x in recommends_knn]
        caching.clear_cache()
        result = SaveSpotifyPlaylist(recommendedTracks_SPIDs)    

        if result != 'Error':
            components.iframe(src="https://open.spotify.com/embed/playlist/5kh3mdIxqKQqmRRtULXftI", width=700, height=90)
            st.dataframe(df_recommends, width=700, height=600)
        else:
            st.dataframe(df_recommends, width=700, height=600)

if __name__ ==  "__main__":
     main()