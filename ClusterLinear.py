import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale

class AggregateClusterLinear(BaseEstimator, RegressorMixin):
    """Applies many cluster linears to a dataset. """
    def __init__(self, clusterFnc=None, regularization_alpha=None):
        if clusterFnc is None:
            self.cf = np.sqrt
        if regularization_alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = regularization_alpha

    def fit(self, X, y, split_on_=0):
        self.split_on = split_on_
        self.models = {}
        unique_indices = np.unique(X[:, self.split_on])
        for i in unique_indices:
            self.models[i] = ClusterLinear()
            Xpart = X[np.where(X[:, self.split_on] == i)] 
            ypart = y[np.where(X[:, self.split_on] == i)]
            self.models[i].clus.n_clusters = int(self.cf(Xpart.shape[0]))
            self.models[i].reg.alpha = self.alpha 
            self.models[i].fit(Xpart[:, np.lib.setdiff1d(np.arange(Xpart.shape[1]),[self.split_on])], ypart)

    def predict(self, X):
        X = X.copy()
        #attach sentinel variables to the array so they can be sorted later
        X = np.concatenate((X, np.arange(X.shape[0]).reshape((X.shape[0],1))), axis=1)
        unique_indices = np.unique(X[:, self.split_on])
        y_list = []
        for i in unique_indices:
            Xpart = X[np.where(X[:, self.split_on] == i)][:,:-1]
            indXpart = X[np.where(X[:, self.split_on] == i)][:,-1]
            y = self.models[i].predict(Xpart[:, np.lib.setdiff1d(np.arange(Xpart.shape[1]),[self.split_on])])
            ind_with_y = np.concatenate((indXpart.reshape((indXpart.shape[0],1)),\
                                        y.reshape((y.shape[0],1))), axis=1)
            y_list.append(ind_with_y)
        y = np.concatenate(y_list)
        y = y[np.argsort(y[:,0])]
        return y[:,1]

class ClusterLinear(BaseEstimator, RegressorMixin):
    """Attempts to cluster data before regression"""
    def __init__(self, Clusterer='kmeans', Regressor='ridge'):
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
        elif self.Regressor == 'linear':
            self.reg = LinearRegression()

    def fit(self, X, y):
        """
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
        elif self.Regressor == 'linear':
            self.reg = LinearRegression()
        """

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
