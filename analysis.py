import pandas as pd
import numpy as np
import sys

import ClusterLinear
from db_to_df import get_df_from_db
from models import artist_df_to_ts_array

def analyze(n_artists = 500, min_works=50, labels=['artistID', 'auctionDate', 'date'], y_label='priceUSD', verbose=True):
    s = []
    if verbose:
        print 'Building dataframe...',
        sys.stdout.flush()

    df = get_df_from_db(n_artists, min_works=min_works)
    df.set_index('_id', inplace=True)
    if verbose:
        print 'done. %d records.' % df.shape[0]
        sys.stdout.flush()
        print 'Converting dataframe to array...',
        sys.stdout.flush()

    for i in df.artistID.value_counts().index:
        s.append(artist_df_to_ts_array(df, i, X_labels=labels, y_label='priceUSD'))
    
    if verbose:
        print 'done.'
        sys.stdout.flush()
    x = pd.concat(s).dropna()

    X = x.values[:,:-1]
    y = x.values[:,-1]
    
    if verbose:
        print 'Fitting model...',
        sys.stdout.flush()

    acl = ClusterLinear.AggregateClusterLinear()

    acl.fit(X, y)
 
    if verbose:
        print 'done.'
        sys.stdout.flush()
    
    cluster_list = []
    for artist in x.artistID.value_counts().index:
        cluster_prediction = acl.models[artist].clus.predict(x[x.artistID == artist].values[:,1:-1])
        artIDs = x[x.artistID == artist].index.values
        artistIDvec = artist * np.ones(x[x.artistID==artist].shape[0])

        cluster_list.append(np.array([artistIDvec, artIDs, cluster_prediction]).T)
        
    return acl, np.concatenate(cluster_list)

