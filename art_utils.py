from skimage import img_as_float
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.filter import threshold_otsu
from scipy.misc import imread
from numpy.linalg import norm
from numpy.fft import fft2
from numpy import mean, std,  sum
from itertools import combinations
from pandas import DataFrame

import pymongo
import datetime

def repeated_sales(df, artistname, artname, r2thresh=7000, fftr2thresh=10000, IMAGES_DIR='/home/ryan/asi_images/'):
    """
        Takes a dataframe, artistname and artname and tries to decide, via image matching, if there is a repeat sale. Returns a dict of lot_ids, each entry a list of repeat sales
    """
    artdf = df[(df['artistID']==artistname) & (df['artTitle']==artname)]

    artdf.images = artdf.images.apply(getpath)
    paths = artdf[['_id','images']].dropna()
    id_dict = {}
    img_buffer = {}
    already_ordered = []
    for i, path_i in paths.values:
        id_dict[i] = []
        img_buffer[i] = img_as_float(rgb2gray(resize(imread(IMAGES_DIR + path_i), (300,300))))
        for j, path_j in paths[paths._id != i].values:
            if j > i and j not in already_ordered:
                if j not in img_buffer.keys():
                    img_buffer[j] = img_as_float(rgb2gray(resize(imread(IMAGES_DIR + path_j), (300,300))))
                if norm(img_buffer[i] - img_buffer[j]) < r2thresh and\
                        norm(fft2(img_buffer[i]) - fft2(img_buffer[j])) < fftr2thresh:
                    id_dict[i].append(j)
                    already_ordered.append(j)
    for key in id_dict.keys():
        if id_dict[key] == []:
            id_dict.pop(key)
    return id_dict

def getpath(x):
    if len(x) > 0:
        return x[0]['path']
    else:
        return None

def image_compare(df, IMAGES_DIR='/home/ryan/asi_images/'):
    '''
    takes a list of n image ids and returns sum(n..n-1) n comparisons of r2 difference, r2(fft) difference, and average number of thresholded pixels
    '''
    img_buffer = {}
    return_list = []
    artdf = df[['_id', 'images']].copy()
    artdf.images = artdf.images.apply(getpath) 
    paths = artdf[['_id','images']].dropna()
    paths.index = paths._id
    paths = paths.images
    if paths.shape[0] < 2:
        return DataFrame([])
    for id_pair in combinations(paths.index, 2):
        if id_pair[0] in img_buffer:
            img1 = img_buffer[id_pair[0]]
        else:
            img_buffer[id_pair[0]] = img_as_float(rgb2gray(resize(imread(IMAGES_DIR + paths[id_pair[0]]), (300,300))))
            img1 = img_buffer[id_pair[0]]
        
        if id_pair[1] in img_buffer:
            img2 = img_buffer[id_pair[1]]
        else:
            img_buffer[id_pair[1]] = img_as_float(rgb2gray(resize(imread(IMAGES_DIR + paths[id_pair[1]]), (300,300))))
            img2 = img_buffer[id_pair[1]]
        return_list.append(
                [id_pair[0], id_pair[1], \
                    norm(img1 - img2), \
                    norm(fft2(img1) - fft2(img2)), \
                    #mean([sum(img1 > threshold_otsu(img1)), sum(img2 > threshold_otsu(img2))])]
                    #mean([sum(img1 > 0.9), sum(img2 > 0.9)])] 
                    std(img1)+std(img2)/2.]
       )
    return DataFrame(return_list, columns=['id1','id2','r2diff', 'fftdiff', 'stdavg'])

def image_compare_from_db(database='asi_database', source_collection='asi_collection', target_collection='duplicate_images', IMAGES_DIR='/home/ryan/asi_images/'):
    client = pymongo.MongoClient()
    db = client[database]
    
    c = db[source_collection]
    c_target = db[target_collection]

    artistIDs = pymongo.cursor.Cursor.distinct(c.find({}), 'artistID')
    
    for ID in artistIDs:
        df = DataFrame(list(c.find({'artistID':ID})))
        title_counts = df['artTitle'].value_counts()
        title_counts = title_counts[title_counts > 1]
        for title in title_counts.index:
            ret = image_compare(df[df['artTitle'] == title], IMAGES_DIR = IMAGES_DIR)
            if ret.shape[0] > 0:
                json_out = eval(image_compare(df[df['artTitle'] == title], IMAGES_DIR = IMAGES_DIR).to_json(orient='records').replace('null', 'None'))
                for rec in json_out:
                    rec['id1'] = int(rec['id1'])
                    rec['id2'] = int(rec['id2'])

                c_target.insert(json_out)
           
def main():
    image_compare_from_db()

if __name__ == '__main__':
    main()
