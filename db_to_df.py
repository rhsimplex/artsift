import pymongo
import pandas as pd
import aggregator_functions
import numpy as np

def get_df_from_db(n_artists, min_works=0, db_name='asi_database', coll_name='asi_collection'):
    client = pymongo.MongoClient()
    db = client[db_name]
    c = db[coll_name]

    artistIDs = np.array(pymongo.cursor.Cursor.distinct(c.find({}), 'artistID'))
    np.random.shuffle(artistIDs)
    sentinel = 0
    i = 0
    dfs = []
    size = artistIDs.shape[0]

    while sentinel < n_artists and i < size:
        if c.find({'artistID':artistIDs[i]}).count() >= min_works:
            dfs.append(pd.DataFrame(list(c.find({'artistID':artistIDs[i]}))))
            sentinel += 1
        i += 1

    return pd.concat(dfs)
