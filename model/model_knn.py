import os
import pickle
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class nneighbors_model(object):
    """
    Implement k nearest neighbors model search
    """
    def __init__(self, model, n_neighbors, metric, sp_playlists, reTrain=False, data_root='./data' ,model_file='kNN.pkl'):
       
        self.model_name = model
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.sp_playlists = sp_playlists
        self.modelfile = os.path.join(data_root, model_file)
        self.defineModel(n_neighbors, metric, reTrain)
    
    def defineModel(self, n_neighbors, metric, reTrain):
        """
        Define nearest neighbors parameters

        :param n_neighbors: number of neighbors to use 
        :param metric: the distance metric to use for the tree  
        :param reTrain: retrain model                      
        """
        if reTrain or not os.path.isfile(self.modelfile):
            self.model = NearestNeighbors(
                n_neighbors=n_neighbors, 
                metric=metric)
            self.fitModel(self.sp_playlists)
        else:
            self.model = pickle.load(open(self.modelfile, 'rb'))
    
    def fitModel(self, sp_playlists):
        """
        Fit the nearest neighbors estimator from the training dataset

        :param sp_playlists: train sparse matrix         
        """   
        self.model.fit(sp_playlists)
        pickle.dump(self.model, open(self.modelfile, 'wb'))
        
    def getNNeighbors(self, X, n_neighbors):
        """
        Returns the first k neighbors for X

        :param X: playlists ID (pid)         
        :param n_neighbors: number of neighbors to use         
        """
        neighbors=self.model.kneighbors(X=X, n_neighbors=n_neighbors+1, return_distance=False)
        # (k + 1) as not to consider its own neighbor , just remove it from neighbors list
        neighbors=neighbors.squeeze().tolist()[:0:-1]
        return neighbors
    
    def getTracksFromNNeighbors(self, neighbors):
        """
        Returns list of tracks from each given neighbor of X

        :param neighbors: neighbors of X         
        """   
        tracksFromNNeighbors=[]
        for x in neighbors:
            tracksFromNNeighbors.append(self.sp_playlists.getrow(x).indices)
        return tracksFromNNeighbors
    
    def getPredictedTracks(self, neighborsTracks, top_k, inputTracks):
        """
        Returns list of top_k predicted tracks from k neighbors track list

        :param neighborsTracks: tracks taken from neighbors of X   
        :param top_k: number of tracks to predict 
        :param inputTracks: tracks in the playlist X                      
        """   
        rTracks = defaultdict(int)
        for i, neighbor in enumerate(neighborsTracks):
            for track in neighbor:
                # measure track by k neighbor rank
                if track not in inputTracks:
                    rTracks[track] +=(1/(i+1)) 
        predictedTracks = sorted(rTracks.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        predictedTracks_ind = [i[0] for i in predictedTracks]
        return predictedTracks_ind
    
    def predict(self, X, top_k, n_neighbors):
        """
        Return the first top_k recommendation of kNearestNeighbors model

        :param X: playlists ID (pid)  
        :param top_k: number of tracks to predict  
        :param n_neighbors: number of neighbors to use                       
        """
        neighbors = self.getNNeighbors(self.sp_playlists.getrow(X), n_neighbors)
        tracksNeighbors = self.getTracksFromNNeighbors(neighbors)
        predictions = self.getPredictedTracks(tracksNeighbors, top_k, list(self.sp_playlists.getrow(X).indices))
        return predictions 

    def predict_random(self, sp_random, top_k, n_neighbors):
        """
        Return the first top_k recommendation of kNearestNeighbors model
        
        :param sp_random: random playlist in sparse matrix format
        :param top_k: number of tracks to predict  
        :param n_neighbors: number of neighbors to use         
        """
        neighbors = self.getNNeighbors(sp_random, n_neighbors)
        tracksNeighbors = self.getTracksFromNNeighbors(neighbors)
        predictions = self.getPredictedTracks(tracksNeighbors, top_k, list(sp_random.getrow(0).indices))
        return predictions          