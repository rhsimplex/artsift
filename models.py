import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcess
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from nltk.corpus import stopwords

def materials_to_array(df, artistID, cutoff=5):
    sw = pd.Series(stopwords.words('english'))
    sw.loc[-1] = 'works'
    words = pd.Series(np.concatenate(df[df.artistID == artistID].materials.str.split().values)).apply(lambda x: x.lower().replace(',',''))
    #common variants
    words.replace('watercolour','watercolor', inplace=True)
    words.replace('india', 'indian', inplace=True)
    goodwords = words[words.isin(sw) == False]
    common_word_count = np.unique(goodwords.values).shape[0]/cutoff
    word_counts = goodwords.value_counts()
    wc = word_counts[word_counts > common_word_count].index

    word_list = df[df.artistID == artistID].materials.str.split().apply(lambda x: map(lambda p: p.lower(),x))

    materials = pd.DataFrame(np.zeros((df[df.artistID == artistID].shape[0],wc.shape[0])), columns = wc, index=df[df.artistID == artistID].index)
    for material in materials.columns:
            for i, val in word_list.iteritems():
                materials[material].ix[i] = material in val
    return materials

def artist_df_to_ts_array(df, artistID, X_labels=['auctionDate', 'date', 'measurements', 'materials'], y_label='priceUSD', halflife = 50):
    '''
    Generates DFs suitable for sklearn (just add .values!) from the given feature labels and target, dropping NAs as applicable
    '''
    artist_price_trend = df[df['artistID']==artistID][X_labels + [y_label]]
    #artist_price_trend = artist_price_trend.dropna()

    if 'auctionDate' in X_labels:
        artist_price_trend['auctionDate'] = pd.to_datetime(artist_price_trend['auctionDate']).astype(np.int64)
    
    DOB = -10000
    DOD = 10000
    
    if 'date' in X_labels:
        #artist_price_trend = artist_price_trend[artist_price_trend['date'] != u'']
        #artist_price_trend[artist_price_trend['date'] == u''] = 'abc'
        artist_price_trend.date = artist_price_trend.date.str.findall('\d{4}').str.get(0)
        #artist_price_trend = artist_price_trend.dropna()
        #artist_price_trend.date = artist_price_trend.date.astype('64')
        try:
            DOB = int(df[df.artistID == artistID]['artistDOB'].iloc[0])
        except ValueError:
            DOB = artist_price_trend.date.mean() - halflife
        try:
            DOD = int(df[df.artistID == artistID]['artistDOD'].iloc[0])
        except ValueError:
            DOD = artist_price_trend.date.mean() + halflife

    #apt = artist_price_trend[(artist_price_trend.date > DOB) & (artist_price_trend.date < DOD)]
    apt = artist_price_trend

    if 'measurements' in X_labels:
        def area(dimlist):
            if len(dimlist) == 4:
                return float(dimlist[1])*float(dimlist[3])
            else:
                return None
        m = apt['measurements'].str.findall('\\d+.\\d+').apply(area)
        m.name = 'area'
        apt = apt.join(m)
        apt.drop('measurements', axis=1, inplace=True)
        apt.rename(columns={'area':'measurements'}, inplace=True)
        #apt.dropna(inplace=True)
    
    if 'materials' in X_labels:
        materials = materials_to_array(df, artistID)
        apt = apt.join(materials)
        apt.drop('materials', axis=1, inplace=True)

    if 'auctionDate' in X_labels:
        apt.sort('auctionDate', inplace=True)
    return apt.astype('float64')

LINEAR_MODELS = {
    'linear_regression': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg', LinearRegression()),
            ]),
        {
            'reg__normalize': [False]
            }
        ),
    
    'logistic_regression': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg', LogisticRegression()),
            ]),
        {
            'reg__C': np.logspace(0.01, 1, 3),
            },
        ),
    'gaussian_process': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg', GaussianProcess()),
            ]),
        {
            'reg__regr': ['constant', 'linear'],
            #'reg__corr': ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear'],
            'reg__corr': ['squared_exponential', 'linear'],
            'reg__nugget': np.finfo(float).eps*np.logspace(1, 15, 2),
            },
        ),
    }

def search_artist_regression(df, artistID, model_name, _verbose=4):
    model, param_grid =LINEAR_MODELS[model_name] 
    X, y = artist_df_to_ts_array(df, artistID)
    gs = GridSearchCV(model, param_grid, verbose=_verbose, scoring='r2')
    gs.fit(X.values, y.values)
    return gs
