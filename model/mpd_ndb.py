import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

class spotify_mpd(object):
    
    def __init__(self, data_root='./data/', 
                       track_file='tracks_ndb.compress', 
                       train_file='train_data_subset_mpdNDB.compress', 
                       test_file='test_data_subset_mpdNDB.compress',
                       train_sm_file='train_subset_mpdNDB_sparse_matrix.npz', 
                       test_sm_file='test_subset_mpdNDB_sparse_matrix.npz',
                       pid_dict_file='old_pid_index.pkl',                 
                       tid_dict_file='old_tid_index.pkl'):            
           
        self.data_root = data_root
        self.tracks_file = track_file
        self.train_file = train_file
        self.test_file = test_file
        self.train_sm_file = train_sm_file
        self.test_sm_file = test_sm_file 
        self.pid_dict_file = pid_dict_file         
        self.tid_dict_file = tid_dict_file        
        
    def read_data(self, file, verbose=False):
        """
        Read file returning a pandas dataframe
        :param file: file path
        :param verbose: print the first rows of the data
        """
        datafile = os.path.join(self.data_root, file)
        data = pd.read_pickle(datafile, compression='gzip')
        if verbose:
            print(data.head())
        return data

    def read_matrix(self, file, verbose=False):
        """
        Read file returning an sparse matrix
        :param file: file path
        :param verbose: print matrix shape
        """
        datafile = os.path.join(self.data_root, file)
        matrix = sp.load_npz(datafile)  
        if verbose:
            print(matrix.shape)           
        return matrix 
    
    def tracks(self, verbose=False):
        """
        Train data
        :param verbose: print track size        
        """
        self.tracks = {}
        self.tracks = self.read_data(self.tracks_file, verbose=False)
        self.n_tot_tracks = self.tracks.tid.unique().shape[0]
        self.n_tot_artist = self.tracks.artist_uri.unique().shape[0]
        self.tracks = self.tracks.filter(['tid', 'spid', 'track_name', 'artist_name', 'album_name', 'duration_ms'])
        if verbose:
            print('Tracks summary: There are %s tracks from %s artists' \
                  %(self.n_tot_tracks, self.n_tot_artist))
    
    def create_dicts(self, verbose=False):
        """
        Build track dict
        :param verbose: print dictionaries size       
        """
        self.tracks_to_tidx = {}
        self.tracks_to_spidx = {}
        self.spIDTotIDMap = {} 
        self.new_pid_dict = {}        
        self.new_tid_dict = {}
        
        track_title = self.tracks['track_name'].map(str) + " - " + self.tracks['artist_name']
        self.tracks_to_tidx = pd.Series(track_title.values, index=self.tracks.tid).to_dict()
        self.tracks_to_spidx = pd.Series(track_title.values, index=self.tracks.spid).to_dict()
        self.spIDTotIDMap = self.tracks.set_index('spid').tid  
        pid_dict_file = os.path.join(self.data_root, self.pid_dict_file)
        self.old_pid_dict = pd.read_pickle(pid_dict_file)         
        tid_dict_file = os.path.join(self.data_root, self.tid_dict_file)
        self.old_tid_dict = pd.read_pickle(tid_dict_file)       
        if verbose:
            print('Track Dictionaries summary: There are %s unique tracks in the dictionaries per tid and spid' \
                  %(len(self.tracks_to_tidx.keys())))   
    
    def train(self, verbose=False):
        """
        Train data
        :param verbose: print train set size         
        """
        self.train = {}
        self.train = self.read_data(self.train_file, verbose)
        self.n_playlists = self.train.pid.unique().shape[0]
        self.n_tracks = self.train.tid.unique().shape[0]
        self.n_rates = self.train.shape[0]
        if verbose:
            print('Train summary: There are %s playlist, %s tracks and %s rates in the train set' \
                  %(self.n_playlists, self.n_tracks, self.n_rates))  
            
    def test(self, verbose=False):
        """
        Test data
        :param verbose: print test set size         
        """
        self.test = {}
        self.test = self.read_data(self.test_file, verbose)
        if verbose:
            print('Test summary: There are %s playlist, %s tracks and %s rates in the test set' \
                  %(self.test.pid.unique().shape[0], self.test.tid.unique().shape[0], 
                    self.test.shape[0]))  
            
    def train_sparse_matrix(self, verbose=False):
        """
        Train sparse matrix
        :param verbose: print train matrix shape         
        """
        self.train_sparse_matrix = self.read_matrix(self.train_sm_file, verbose)
        self.n_sm_playlists = self.train_sparse_matrix.shape[0]
        self.n_sm_tracks = self.train_sparse_matrix.shape[1]
        self.n_sm_rates = self.train_sparse_matrix.nnz
        if verbose:
            print('Train Matrix summary: %sx%s sparse matrix of dtype %s ' \
                  'with %s stored elements in Compressed Sparse Row format' \
                  %(self.train_sparse_matrix.shape[0], self.train_sparse_matrix.shape[1], 
                    self.train_sparse_matrix.dtype, self.train_sparse_matrix.nnz)) 
            
    def test_sparse_matrix(self, verbose=False):
        """
        Test sparse matrix
        :param verbose: print test matrix shape        
        """
        self.test_sparse_matrix = self.read_matrix(self.test_sm_file, verbose)
        if verbose:
            print('Test Matrix summary: %sx%s sparse matrix of dtype %s ' \
                  'with %s stored elements in Compressed Sparse Row format' \
                  %(self.test_sparse_matrix.shape[0], self.test_sparse_matrix.shape[1], 
                    self.test_sparse_matrix.dtype, self.test_sparse_matrix.nnz))  

    def get_title(self, tracks_id, translate=True, verbose=False):
        """
        Given a track id, return title
        :param tracks_id: tracks id
        :param translate: True to map with original tid when DF is subsetmpdNDB   
        :param verbose: print titles from list of track ids               
        """
        if translate:
            title = self.tracks_to_tidx[self.old_tid_dict[tracks_id]]                
        else:
            title = self.tracks_to_tidx[tracks_id]
        if verbose:
            print(title)   
        return title               
            
    def get_data(self, verbose=False):
        """
        Load data together
        :param verbose: print executed function details        
        """
        self.tracks(verbose)
        self.create_dicts(verbose)
        self.train(verbose)
        self.test(verbose)
        self.train_sparse_matrix(verbose)
        self.test_sparse_matrix(verbose)    