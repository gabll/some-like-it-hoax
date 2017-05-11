# Functions utils for Exploratory Data Analysis, like filtering
# Author: Marco Della Vedova - marco.dellavedova@gmail.com

import json
import itertools
import numpy as np


with open('hoaxpagedict.json') as pagedatafile:
    hoaxpagedict = json.load(pagedatafile)

pages_hoax = [pageid for pageid in hoaxpagedict.keys() if hoaxpagedict[pageid]]
pages_nonhoax = [pageid for pageid in hoaxpagedict.keys() if not hoaxpagedict[pageid]]


def filter_post(datadict, min_likes):
    dataset_filtered = {post_id: likes for (post_id, likes) in datadict.items() if len(likes) >= min_likes}
    return dataset_filtered


def get_all_likes(datadict):
    return list(itertools.chain(*datadict.values()))


def get_user_set(datadict):
    return set(get_all_likes(datadict))


def filter_user(datadict, min_likes):
    from collections import Counter
    likes = get_all_likes(datadict)
    user_count = Counter(likes)
    user_set_f = {u for (u, c) in user_count.items() if c >= min_likes}
    posts_susers = dict()  # Posts and *selected* users
    for (post_id, likes) in datadict.items():
        newlikes = [x for x in likes if x in user_set_f]
        posts_susers[post_id] = newlikes
    return posts_susers


def get_number_of_users(datadict):
    return len(get_user_set(datadict))


def get_number_of_likes(datadict):
    return len(get_all_likes(datadict))


def is_hoax(postid, hoaxpages=pages_hoax):
    return postid.split('_')[0] in hoaxpages


# Split likes into 'likes to hoax posts' and 'likes to non-hoax posts'
# Note: I used pre-allocation for speed-up the precess
def split_likes(data):
    initial_dim = get_number_of_likes(data)
    likes_hoax = np.zeros(initial_dim, dtype='object')
    likes_nhoax = np.zeros(initial_dim, dtype='object')
    lasth = 0
    lastn = 0
    for (postid, likes) in data.items():
        if len(likes) > 0:
            if is_hoax(postid):
                # likes_hoax = likes_hoax + likes
                likes_hoax[lasth:lasth+len(likes)] = likes
                lasth += len(likes)
            else:
                # likes_nhoax = likes_nhoax + likes
                likes_nhoax[lastn:lastn+len(likes)] = likes
                lastn += len(likes)
    likes_hoax = likes_hoax[0:lasth].tolist()
    likes_nhoax = likes_nhoax[0:lastn].tolist()
    return likes_hoax, likes_nhoax