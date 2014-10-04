import numpy as np
import re
import pymongo
from nltk.corpus import stopwords

def area(dimlist):
    if len(dimlist) == 4:
        return float(dimlist[1].replace(',',''))*float(dimlist[3].replace(',',''))
    else:
        return None


def tag_to_tags(string, omitted_words):
    '''
        omitted_words should be a set, not list
    '''
    return list( set(string.lower().split()) - omitted_words )

def add_tags(json_entry, omitted_words):
    json_entry['note_tags'] = tag_to_tags(json_entry['lotNote'], omitted_words)
    json_entry['material_tags'] = tag_to_tags(json_entry['materials'], omitted_words)
    try:
        json_entry['area'] = area(re.findall('\\d+.\\d+', json_entry['measurements']))
    except ValueError:
        json_entry['area'] = None

    return json_entry

def iterate_over_db(db_name='asi_database', coll_name='asi_collection'):
    client = pymongo.MongoClient()
    db = client[db_name]
    c = db[coll_name]

    curs = c.find({})

    omitted_words = set(stopwords.words('english'))

    while curs.alive:
        c.save(add_tags(curs.next(), omitted_words))
