# ndb-music-recommender
#### Repository for NDB Music Festival Recommender project from KSchool - Madrid, Master degree in Data Science - July 2021
<br>
<p>This project seeks to generate a Music Recommender System 
whose target user would be any attendee to Live Music Festival, Noches del Botánico (NDB).</p>
<p>NDB festival is characterized by the eclecticism 
of its musical commitment where in each edition all musical genres coexist, 
from flamenco to jazz, rock to blues through urban rhythms, electronics or any avant-garde music.</p>

**Data Acquisition**

<p>To collect the data to build the model I have used Spotify API 
and public dataset, Spotify Million Playlist Dataset</p>

> **Spotify Account** <br>
In order to use Spotify API, you´ll need an account (free or paid).<br>
Follow Spotify guide to create one. [App Settings](https://developer.spotify.com/documentation/general/guides/app-settings/)
<br>

> **settings.env file** <br> 
In order to not uploading your Spotify settings to Github, you can create a .env text file and place it into your local Github repository. Create a .gitignore file at the root folder of your project so the .env file ('../settings/') is not uploaded to the remote repository. The content of the .env text file should look like this:  

>  {
>>   "SPOTIPY_CLIENT_ID": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
>>   "SPOTIPY_CLIENT_SECRET": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  
>>   "SPOTIPY_REDIRECT_URI": "http://xxxxxxxxxxxxxx",  
>>   "SPOTIPY_USER": "xxxxxx",  
>>   "PLAYLIST_ID": "xxxxxxxxxxxxxxxxxxxxxxx"  
> }

> **Spotify Million Playlist Dataset** <br>
The [MPD dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) can be downloaded by registered participants from the [Resources](https://www.aicrowd.com/participants/sign_in) page of AIcrowd, download file: spotify_million_playlist_dataset.zip (5.39 GB).<br>
Unzip json files and copied  at the root folder of your project in the path '..
/data/playlists'.
<br>
