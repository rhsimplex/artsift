import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.gaussian_process import GaussianProcess
from sklearn.pipeline import Pipeline
def artist_df_to_ts_array(df, artistID, X_labels=['auctionDate', 'date'], y_label='priceUSD', halflife = 50):
    '''
    Generates DFs suitable for sklearn (just add .values!) from the given feature labels and target, dropping NAs as applicable
    '''
    artist_price_trend = df[df['artistID']==artistID][X_labels + [y_label]]
    artist_price_trend = artist_price_trend.dropna()

    if 'auctionDate' in X_labels:
        artist_price_trend['auctionDate'] = pd.to_datetime(artist_price_trend['auctionDate']).astype(np.int64)
    
    DOB = -10000
    DOD = 10000
    
    if 'date' in X_labels:
        artist_price_trend = artist_price_trend[artist_price_trend['date'] != u'']
        artist_price_trend.date = artist_price_trend.date.str.findall('\d{4}').str.get(0)
        artist_price_trend = artist_price_trend.dropna()
        artist_price_trend.date = artist_price_trend.date.astype('int64')
        try:
            DOB = int(df[df.artistID == artistID]['artistDOB'].iloc[0])
        except ValueError:
            DOB = artist_price_trend.date.mean() - halflife
        try:
            DOD = int(df[df.artistID == artistID]['artistDOD'].iloc[0])
        except ValueError:
            DOD = artist_price_trend.date.mean() + halflife

    apt = artist_price_trend[(artist_price_trend.date > DOB) & (artist_price_trend.date < DOD)]
    return apt[X_labels].astype('float64'), apt[y_label]

MODELS = {
    'linear_regression': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg', LinearRegression()),
            ]),
        ),
    
    'logistic_regression': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg', LogisticRegression()),
            ]),
        {
            'reg__C': [0.01, 0.1, 1.0, 10.],
            },
        ),
    'gaussian_process': (
        Pipeline([
            ('scale', StandardScaler()),
            ('reg' GaussianProcess),
            ]),
        {
            'reg__regr': ['constant', 'linear', 'quadratic'],
            'reg__corr': ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear'],
            },
        ),
    }
