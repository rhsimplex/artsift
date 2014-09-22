from skimage import img_as_float
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import img_as_float
from scipy.misc import imread
from numpy.linalg import norm
from numpy.fft import fft2

def repeated_sales(df, artistname, artname, r2thresh=7000, fftr2thresh=10000, IMAGES_DIR='/home/ryan/asi_images/'):
    """
        Takes a dataframe, artistname and artname and tries to decide, via image matching, if there is a repeat sale. Returns a dict of lot_ids, each entry a list of repeat sales
    """
    artdf = df[(df['artistID']==artistname) & (df['artTitle']==artname)]
    def getpath(x):
        if len(x) > 0:
            return x[0]['path']
        else:
            return None

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
