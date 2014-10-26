import pandas as pd
import numpy as np
import sys

import ClusterLinear
from db_to_df import get_df_from_db
from models import artist_df_to_ts_array
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def analyze(n_artists = 500, min_works=50, labels=['artistID', 'auctionDate', 'date'], y_label='priceUSD', verbose=True, daterange = ('2001-01-01', '2014-01-01'), db_name='asi_database'):
    s = []
    if verbose:
        print 'Building dataframe...',
        sys.stdout.flush()
    x = None
    df = get_df_from_db(n_artists, min_works=min_works, db_name=db_name)
    df.set_index('_id', inplace=True)
    ad = pd.to_datetime(df.auctionDate)
    df = df[(ad > daterange[0]) & (ad < daterange[1])]

    if verbose:
        print 'done. %d records.' % df.shape[0]
        sys.stdout.flush()
        print 'Converting dataframe to array...',
        sys.stdout.flush()

    for i in df.artistID.value_counts().index:
        try:
            if x is None:
                x = artist_df_to_ts_array(df, i, X_labels=labels, y_label='priceUSD', n_tags=50)
            else:
                x = pd.concat([x, artist_df_to_ts_array(df, i, X_labels=labels, y_label='priceUSD', n_tags=50)] )
        except KeyError:
            pass ####
            
    if verbose:
        print 'done.'
        sys.stdout.flush()
    #x = pd.concat(s).dropna()
    x.dropna(inplace=True)
    #move y_label to last position
    l = x.columns.tolist()
    l.append(l.pop(l.index(y_label)))
    x = x[l]

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
        # predicted cluster
        cluster_prediction = acl.models[artist].clus.predict(x[x.artistID == artist].values[:,1:-1])
        
        # unique auction ID of prediction
        artIDs = x[x.artistID == artist].index.values
        
        # artistID (all the same) 
        artistIDvec = artist * np.ones(x[x.artistID==artist].shape[0])
        
        #merge together
        cluster_list.append(np.array([artistIDvec, artIDs, cluster_prediction]).T)
        

        cluster_list[-1] = np.concatenate([cluster_list[-1], np.apply_along_axis(lambda x: acl.scores[x[0]][x[2]], 1, cluster_list[-1])], axis=1)
    
    
    dfresults = pd.DataFrame(np.concatenate(cluster_list), columns=['artistID', '_id','cluster','n','r2'])
    dfresults['time_r2'] = 0.
    dfresults['coef'] = 0.
    dfresults.set_index('_id', inplace=True)

    rid = Ridge()
    for artistID in np.unique(dfresults.artistID.values):
        for clusterID in np.unique(dfresults[dfresults.artistID == artistID].cluster):
            df_temp = df.ix[dfresults[(dfresults.artistID == artistID) & (dfresults.cluster == clusterID)].index.values][['auctionDate', 'priceUSD']]
            X = pd.to_datetime(df_temp.auctionDate).astype('int').values
            X = X.reshape((X.shape[0], 1))
            y = df_temp.priceUSD
            rid.fit(X, y)
            coef = rid.coef_[0]
            r2 = r2_score(y, rid.predict(X))
            dfresults.time_r2.ix[df_temp.index.values] = r2
            dfresults.coef.ix[df_temp.index.values] = coef

    return dfresults

def main():
    dfresults = analyze(n_artists=15, db_name = 'asi-database', daterange = ('2001-01-01', '2014-01-01'), labels=['artistID', 'date','area','material_tags'])
    dfresults.to_csv('test.csv')

    print 'Test success!'
    
    dfresults = analyze(n_artists=300000, db_name = 'asi-database', daterange = ('2013-01-01', '2014-01-01'), labels=['artistID', 'date','area','material_tags'])
    dfresults.to_csv('results_2013_2014.csv')
    dfresults = analyze(n_artists=300000, db_name = 'asi-database', daterange = ('2005-01-01', '2014-01-01'), labels=['artistID', 'date','area','material_tags'])
    dfresults.to_csv('results_2005_2014.csv')
    dfresults = analyze(n_artists=300000, db_name = 'asi-database', daterange = ('2001-01-01', '2014-01-01'), labels=['artistID', 'date','area','material_tags'])
    dfresults.to_csv('results_2001_2014.csv')

if __name__ == '__main__':
    main()
