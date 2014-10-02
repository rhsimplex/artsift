import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale

class ClusterLinear(BaseEstimator, RegressorMixin):
    """Attempts to cluster data before regression"""
    def __init__(self, Clusterer='kmeans', Regressor='linear'):
        self.Clusterer = Clusterer
        self.Regressor = Regressor
        self.reg_models = {}
        
        if self.Clusterer == 'affinity':
            self.clus = AffinityPropagation()
        elif self.Clusterer == 'meanshift':
            self.clus = MeanShift()
        elif self.Clusterer == 'kmeans':
            self.clus = KMeans()
                
        if self.Regressor == 'ridge':
            self.reg = Ridge()
        elif self.Regressor == 'randomforest':
            self.reg = RandomForestRegressor()
        elif self.Regressor == 'lasso':
            self.reg = Lasso()
        elif self.Regressor == 'randomforest':
            self.reg = RandomForestRegressor()
        elif self.Regressor == 'linear':
            self.reg = LinearRegression()

    def fit(self, X, y):
        if self.Clusterer == 'affinity':
            self.clus = AffinityPropagation()
        elif self.Clusterer == 'meanshift':
            self.clus = MeanShift()
        elif self.Clusterer == 'kmeans':
            self.clus = KMeans()
                
        if self.Regressor == 'ridge':
            self.reg = Ridge()
        elif self.Regressor == 'randomforest':
            self.reg = RandomForestRegressor()
        elif self.Regressor == 'lasso':
            self.reg = Lasso()
        elif self.Regressor == 'randomforest':
            self.reg = RandomForestRegressor()
        elif self.Regressor == 'linear':
            self.reg = LinearRegression()
        

        self.reg_models = {}

        y_cluster = scale(y)
        
        self.clus.fit(np.concatenate((X, y_cluster.reshape((1,y_cluster.shape[0])).T), axis=1))
        self.clus.cluster_centers_ = self.clus.cluster_centers_[:,:-1]
        
        for label in np.unique(self.clus.labels_):
            self.reg_models[label] = clone(self.reg)
            self.reg_models[label].fit(X[np.where(self.clus.labels_ == label)], y[np.where(self.clus.labels_ == label)])
        return self

    def predict(self, X):
        class_labels = self.clus.predict(X)
        y_pred = np.zeros(class_labels.shape[0])
        for label in np.unique(class_labels):
            indices = np.where(class_labels == label)
            y_pred[indices] = self.reg_models[label].predict(X[indices])
        return y_pred
