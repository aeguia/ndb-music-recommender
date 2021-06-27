import os
import re
import ijson
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp


def parseURI(uri):
    """
    Return Spotify ID (track, artist or album) from Spotify URI
    :param uri: Spotify uri
    """    
    return uri.split(':')[2]


def GetTracksIDs(tracks, spIDTotIDMap):
    """
    Return a list of Tracks IDs from a list of track Spotify IDs
    :param tracks: list of track Spotify IDs
    :param spIDTotIDMap: Map dictionary from track Spotify ID to Track ID    
    """     
    l_tracksIDs = [spIDTotIDMap.get(i) for i in tracks]
    return l_tracksIDs


def ReadData(playlist_root, data_root, playlist_file, tracks_file, numfiles=1000, l_artist_uri=[]):
    """
    Read MPD json datafiles to Playlists and Tracks dataframe and save
    :param playlist_root: json datafiles path
    :param data_root: data files path
    :param playlist_file: playlists filename
    :param tracks_file: tracks filename
    :param numfiles: number of json datafiles   
    :param list_ndb: list of NDB spotify artists uri       
    """
    # Valid column names
    playlist_cols = ['pid', 'name', 'num_tracks', 'num_artists', 'num_albums', 'num_followers', 'duration_ms','tracks']
    tracks_cols = ['track_uri', 'track_name', 'duration_ms', 'artist_uri', 'artist_name', 'album_uri', 'album_name']
    
    tracksAdded = set()
    l_playlists = []
    l_tracks = []

    ndb_playlist = False

    files = [f for f in os.listdir(playlist_root) if os.path.isfile(os.path.join(playlist_root, f))]
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    for i, file in tqdm(enumerate(files[0:int(numfiles)])):

        datafile = os.path.join(playlist_root, file) 
        with open(datafile, 'r') as f:

            playlists = ijson.items(f, 'playlists.item')

            # If NDB Artists list is empty, read input num files complete
            if not l_artist_uri:              

                for playlist in playlists:               

                    for track in playlist['tracks']:                   

                        if parseURI(track['track_uri']) not in tracksAdded:
                            l_tracks.append([track[col] for col in tracks_cols])
                            tracksAdded.add(parseURI(track['track_uri']))

                    # List of tracks containing Spotify IDs instead of whole track object
                    playlist['tracks'] =  [parseURI(x['track_uri']) for x in playlist['tracks']]
                    l_playlists.append([playlist[col] for col in playlist_cols])

                print('File Read Number: ' + str(i))

            else: 

                for playlist in playlists: 

                    for track in playlist['tracks']: 

                        if track['artist_uri'] in l_artist_uri:
                            ndb_playlist=True
                            break

                    if ndb_playlist:

                        for track in playlist['tracks']: 

                            if parseURI(track['track_uri']) not in tracksAdded:
                                l_tracks.append([track[col] for col in tracks_cols])
                                tracksAdded.add(parseURI(track['track_uri']))
                        # List of tracks containing Spotify IDs instead of whole track object
                        playlist['tracks'] =  [parseURI(x['track_uri']) for x in playlist['tracks']]
                        l_playlists.append([playlist[col] for col in playlist_cols]) 
                        ndb_playlist = False                     

                print('File Read Number: ' + str(i))
                
    # Create Dataframes of Playlists and Tracks
    playlists_df = pd.DataFrame(l_playlists, columns=playlist_cols)  
    if l_artist_uri:
        playlists_df['pid'] = playlists_df.index 
    print('Created playlists_df')
    tracks_df = pd.DataFrame(l_tracks, columns=tracks_cols)
    print('Created tracks_df')
    
    # Add spid - string id and tid - numeric id for tracks
    tracks_df['spid'] = tracks_df.apply(lambda row: parseURI(row['track_uri']), axis=1)
    print('Created tracks_df - spid')
    tracks_df['tid'] = tracks_df.index
    
    # Write DFs to Pickle GZIP file
    print(f'Export DF {len(playlists_df)} playlists to Pickle')
    p_file = os.path.join(data_root, playlist_file)
    playlists_df.to_pickle(p_file, compression='gzip')  
    print(f'Export DF {len(tracks_df)} tracks to Pickle')
    t_file = os.path.join(data_root, tracks_file)
    tracks_df.to_pickle(t_file, compression='gzip')


def BuildPlaylistRatings(playlists, tracks, data_root, matrix_file):
    """
    Build sparse matrix for playlist-track matrix
    :param playlists: playlist df
    :param tracks: tracks df
    :param data_root: data files path
    :param matrix_file: sparse matrix filename      
    """
    spIDTotIDMap = tracks.set_index('spid').tid

    playlistIDs = list(playlists["pid"])

    print('Create sparse matrix mapping playlists to tracks')
    # Create the matrix with initial shape (M,N) dtype np.int8
    playlistTrackRating = sp.dok_matrix((len(playlistIDs), len(spIDTotIDMap)), dtype=np.int8)

    for i in tqdm(range(len(playlistIDs))):
        
        # Get playlist_ID and track_spIDs
        playlistID = playlistIDs[i]
        # List of tracks per paylist
        trackID = playlists.loc[playlistID]['tracks']

        playlistIDX = playlistID        
        # Get track_ID instead track_spID
        trackIDX = [spIDTotIDMap.get(i) for i in trackID]
        
        # Set index to 1 if playlist has song
        playlistTrackRating[playlistIDX, trackIDX] = 1 
        
    #        print('Playlist Rates: ' + str(i))

    ratings_matrix_file = os.path.join(data_root, matrix_file)
    sp.save_npz(ratings_matrix_file, playlistTrackRating.tocsr())