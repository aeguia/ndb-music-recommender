import pandas as pd
import numpy as np
import random
import scipy.sparse as sp

def GenerateRandomNDBPlaylist(df, year=0):
    """
    Return a ramdom playlist of track IDs
    :param df: dataframe
    :param year: NDB edition year, by default 0 equals All    
    """     
    if year==0:
        playlist = df.groupby('artist_name')['tid'].sample().to_list()
        random.shuffle(playlist)
    elif year==2016:
        df_2016 = df[df['2016'] == 1]
        playlist = df_2016.groupby('artist_name')['tid'].sample().to_list()
        random.shuffle(playlist)
    elif year==2017:
        df_2017 = df[df['2017'] == 1]
        playlist = df_2017.groupby('artist_name')['tid'].sample().to_list() 
        random.shuffle(playlist)
    elif year==2018:
        df_2018 = df[df['2018'] == 1]
        playlist = df_2018.groupby('artist_name')['tid'].sample().to_list()  
        random.shuffle(playlist) 
    elif year==2019:
        df_2019 = df[df['2019'] == 1]
        playlist = df_2019.groupby('artist_name')['tid'].sample().to_list()  
        random.shuffle(playlist) 
    elif year==2021:
        df_2021 = df[df['2021'] == 1]
        playlist = df_2021.groupby('artist_name')['tid'].sample().to_list() 
        random.shuffle(playlist)  
    else:
        print(f'Error. Introduce one NDB edition (2016, 2017, 2018, 2019, 2021) or 0 to select All')
        return
    return playlist

def BuildSparsePlaylist(playlist, sp_len):
    """
    Return a sparse matrix of 1 row
    :param playlist: list of track IDs
    :param sp_len: sp matrix column size   
    """      
    sp_playlist = sp.dok_matrix((1, sp_len), dtype=np.int8)
    sp_playlist[0,playlist] = 1
    return sp_playlist.tocsr()

def memory_usage(df):
    """
    Return the memory usage of a dataframe in Mb
    :param df: dataframe
    """     
    return(round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2))

def optimize_floats(df):
    """
    Downcast float columns to the smallest possible float datatype
    :param df: dataframe
    """      
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df

def optimize_ints(df):
    """
    Downcast int columns to the smallest possible int datatype
    :param df: dataframe
    """    
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def optimize(df):
    """
    Downcast numeric columns to the smallest possible datatype
    :param df: dataframe
    """      
    return optimize_floats(optimize_ints(df))
