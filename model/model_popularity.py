class base_model(object):
    """
    Implement popularity model: Most rated track search
    """
    def __init__(self, model, train_df):
        self.model_name = model
        self.train_df = train_df
        self.model = self.mostRated()
    
    def mostRated(self):
        """
        Most rated tracks
        """
        return self.train_df.groupby('tid')['pid'].count().sort_values(ascending=False)
        
    def predict(self, X, top_k):
        """
        Returns the first top_k recommendation of the popularity model

        :param X: playlists ID (pid)  
        :param top_k: number of tracks to predict        
        """
        return self.model.index.values[:top_k]       