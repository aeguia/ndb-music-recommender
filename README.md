## ndb-music-recommender

*This is the Repository for NDB Music Festival Recommender, my end of master´s project in KSchool - Madrid, Master degree in Data Science - July 2021*

This project seeks to generate a Music Recommender System whose target user would be any attendee to Live Music Festival, Noches del Botánico (NDB).</p>NDB festival is characterized by the eclecticism of its musical commitment where in each edition all musical genres coexist, from flamenco to jazz, rock to blues through urban rhythms, electronics or any avant-garde music.</p>

<div style="background-color:#e6b945">The present work is an academic project, not professional. Instructions for installing and executing the project are detailed here at README. All details concerning the Machine Learning Model development process can be found at TFM Memory - NDB Music Recommender document.</div>

#### Get Data

To collect the data to build the model I have used Spotify API and public dataset, Spotify Million Playlist Dataset</p>

> **Spotify Account**
In order to use Spotify API, you´ll need an account (free or paid).  
Follow Spotify guide to create one. [App Settings](https://developer.spotify.com/documentation/general/guides/app-settings/)
>
> **settings.env file**  
In order to not uploading your Spotify settings to Github, you can create a .env text file and place it into your local Github repository. Create a .gitignore file at the root folder of your project so the .env file ('../settings/') is not uploaded to the remote repository. The content of the .env text file should look like this:
>
>{  
>>"SPOTIPY_CLIENT_ID": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
>>"SPOTIPY_CLIENT_SECRET": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
>>"SPOTIPY_REDIRECT_URI": "http://xxxxxxxxxxxxxx",  
>>"SPOTIPY_USER": "xxxxxx",  
>>"PLAYLIST_ID": "xxxxxxxxxxxxxxxxxxxxxxx"  
>}
>
> **Spotify Million Playlist Dataset**  
The [MPD dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) can be downloaded by registered participants from the [Resources](https://www.aicrowd.com/participants/sign_in) page of AIcrowd, download file: spotify_million_playlist_dataset.zip (5.39 GB).  
Unzip json files and copied  at the root folder of your project in the path '../data/playlists'.

#### Execution Guide  

To run code i´ve created a Conda Virtual Environment with Python 3.8.5  installing extra libraries/packages detailed in requirements.txt file.

* **NDBArtists_SpotifyDataAdquision.ipynb** (*/data_acquisition*), extracts NDB artists features using Spotify API.  
    ***Input:***
  * *NDB_artist_2021_2016.csv* (*/data*), NDB Festival Concerts from 2019 to 2021.  
  * *settings.env* (*/settings*), Spotify environment variables.

* **ReadInMPDToDFpkl_NDB.ipynb**, parse MPD json files into Playlists and Tracks dataframes and builds co-ocurrence matrix filtering MPD playlists by NDB artists.  
    ***Input:***  
  * *artists_ndb.csv*, processed dataframe.  
  * *mpd.slice.0-999.json - mpd.slice.999000-999999.json*, MPD json files.

* **NDBArtistsFromMPD.ipynb**, extracts the subset of tracks played by NDB artists from MPD_NDB Tracks file.  
    ***Input:***  
  * *artists_ndb.csv*, processed dataframe.
  * *tracks_ndb.compress*, MPD_NDB tracks - processed dataframe.

* **EDA_MPD_NDB.ipynb**, exploratory data analysis of Playlists, Tracks and Rating Matrix from MPD_NDB dataset.
    ***Input:***  
  * *playlists_ndb.compress*, MPD_NDB playlists - processed dataframe.
  * *tracks_ndb.compress*, MPD_NDB tracks processed dataframe.
  * *ndb_mpd_tracks.compress*, NDB artists tracks - processed dataframe.

* **MPDToNDB_subset.ipynb**, dimension reduction of MPD_NDB dataset.  
    ***Input:***  
  * *matrix_playlistTrackRating_ndb.npz*, MPD_NDB ratings - matrix processed.  

* **TrainTestSplit_subset_mpdNDB.ipynb**, train-test split of MPDToNDB_subset.  
    ***Input:***  
  * *subset_mpdNDB_sparse_matrix.npz*, subset of MPD_NDB ratings - matrix processed.  

* **test_popularity_model.ipynb**, train, test and evaluation of popularity Model.  
    ***Input:***  
  * *./model/mpd_ndb.py*, MPD_NDB subset read data class.
  * *./model/model_popularity.py*, popularity model class.
  * *./model/evaluate_model.py*, evaluation metrics class.  

* **test_knn_model.ipynb**, train, test and evaluation of k Nearest Neighbors Model.  
    ***Input:***  
  * *./model/mpd_ndb.py*, MPD_NDB subset read data class.
  * *./model/model_knn.py*, k Nearest Neighbors model class.
  * *./model/evaluate_model.py*, evaluation metrics class.
  * *./results/model_scores.csv*, popularity and kNN models metric scores - processed dataframe.  

* **NDB_Recommend.ipynb**, execute kNN final model fitted with NDB artist playlists by edition using dataset complete MPD_NDB ratings.  
    ***Input:***  
  * *settings.env* (*/settings*), Spotify environment variables.
  * *ndb_mpd_tracks.compress*, NDB artists tracks - processed dataframe.
  * *tracks_ndb.compress*, MPD_NDB tracks processed dataframe.  
  * *matrix_playlistTrackRating_ndb.npz*, MPD_NDB ratings - matrix processed.  

#### NDB Recommender App

The front-end app uses Streamlit app framework and can be executed from root folder of the project.
> streamlit run ndb_recommender.py  

User has to select NDB Edition Year and number of tracks to be recommended and click "Recommend..." button.
