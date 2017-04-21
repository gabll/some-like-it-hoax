import numpy as np
from collections import Counter
from scipy.sparse import dok_matrix, csr_matrix
import itertools


def cut_dataset(data, hoax_pages, min_post_like=10, min_user_like=30, print_results=False):
    """returns the dataset filtered with these parameters:
    min_post_like: post with at least n likes
    min_user_like: users that have given at least n likes
    print_results: if True, prints the filtering effect
    output: sparse like_matrix and page/hoax label columns
    """

    #post filter
    data_posts_f = {post: users
                      for post, users in data.items()
                      if len(users) >= min_post_like}
    if print_results:
        print("Posts with %d+ likes = %d (%d hoax)" %
              (min_post_like,
               len(data_posts_f),
               len([post for post in data_posts_f.keys() if post.split('_')[0] in hoax_pages])))

    #user filter
    user_list = []
    data_users_f = dict()
    for i in data.values():
        user_list.extend(i)
    user_count = Counter(user_list)
    users_filtered = {user for (user, count) in user_count.items() if count >= min_user_like}
    data_users_f = {post: [user for user in users if user in users_filtered]
                         for post, users in data_posts_f.items()}
    if print_results:
        print("Users with %d+ likes = %d" % (min_user_like, len(users_filtered)))

    #postid, userid conversion
    uid2n = dict((user_id, user_n) for (user_n, user_id) in enumerate(users_filtered))
    pid2n = dict((post_id, post_n) for (post_n, post_id) in enumerate(data_users_f.keys()))

    #matrix bulding
    like_matrix = dok_matrix((len(pid2n), len(uid2n)), dtype=np.int8)
    pages = []
    hoaxes = []
    for post, users in data_users_f.items():
        page = post.split('_')[0]
        pages.append(page)
        if page in hoax_pages:
            hoaxes.append(1)
        else:
            hoaxes.append(0)
        for user in users:
            like_matrix[pid2n[post], uid2n[user]] = True
    if print_results:
        nlikes_filtered =  len(like_matrix)
        nposts_filtered = len(pid2n)
        n_users_filtered = len(uid2n)
        lf = nlikes_filtered / (nposts_filtered*n_users_filtered)
        print("%d non-zero values out of %d (loading factor: %.2f%%)" %
              (nlikes_filtered, len(pid2n)*len(uid2n), lf*100))

    return like_matrix, pages, hoaxes


def split_pages(like_matrix, pages, hoaxes, pages_tosplit):
    """splits like_matrix and hoaxes depending if they are or not in pages_tosplit
    output type from dok matrix to csr matrix"""

    like_csr = like_matrix.tocsr()
    bool_split = np.in1d(pages, pages_tosplit)

    ix_split = np.where(bool_split)[0]
    matrix_split = like_csr[ix_split,:]
    hoax_split = [hoaxes[i] for (i, page) in enumerate(pages) if page in pages_tosplit]

    ix_split_not = np.where(np.logical_not(bool_split))[0]
    matrix_split_not = like_csr[ix_split_not,:]
    hoax_split_not = [hoaxes[i] for (i, page) in enumerate(pages) if page not in pages_tosplit]

    return matrix_split, hoax_split, matrix_split_not, hoax_split_not


def get_all_likes(datadict):
    return list(itertools.chain(*datadict.values()))

def get_number_of_likes(datadict):
    return len(get_all_likes(datadict))

def is_hoax(postid, hoaxpages):
    return postid.split('_')[0] in hoaxpages

def split_likes(data, hoaxpages):
    initial_dim = get_number_of_likes(data)
    likes_hoax = np.zeros(initial_dim, dtype='object')
    likes_nhoax = np.zeros(initial_dim, dtype='object')
    lasth = 0
    lastn = 0
    for (postid, likes) in data.items():
        if len(likes) > 0:
            if is_hoax(postid, hoaxpages):
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


def filter_intersection(data, hoax_pages, print_results=False):
    """returns the dataset filtered with only the users who liked at least
    one post belonging to a hoax page and one post belonging to a non-hoax page
    print_results: if True, prints the filtering effect
    output: sparse like_matrix and page/hoax label columns
    """

    likes_hoax, likes_nhoax = split_likes(data, hoax_pages)
    hoax_likers = set(likes_hoax)
    nonhoax_likers = set(likes_nhoax)
    both_likers = nonhoax_likers.intersection(hoax_likers)
    if print_results:
        print('Total users: %d, Intersection users: %d' % (len(set(hoax_likers.union(nonhoax_likers))),len(both_likers)))

    #user filter
    data_users_f = {post: [user for user in users if user in both_likers]
                         for post, users in data.items()
                         if len([user for user in users if user in both_likers])>0}

    #postid, userid conversion
    uid2n = dict((user_id, user_n) for (user_n, user_id) in enumerate(both_likers))
    pid2n = dict((post_id, post_n) for (post_n, post_id) in enumerate(data_users_f.keys()))

    #matrix bulding
    like_matrix = dok_matrix((len(pid2n), len(uid2n)), dtype=np.int8)
    pages = []
    hoaxes = []
    for post, users in data_users_f.items():
        page = post.split('_')[0]
        pages.append(page)
        if page in hoax_pages:
            hoaxes.append(1)
        else:
            hoaxes.append(0)
        for user in users:
            like_matrix[pid2n[post], uid2n[user]] = True
    if print_results:
        nlikes_filtered =  len(like_matrix)
        nposts_filtered = len(pid2n)
        n_users_filtered = len(uid2n)
        lf = nlikes_filtered / (nposts_filtered*n_users_filtered)
        print("%d non-zero values out of %d (loading factor: %.2f%%)" %
              (nlikes_filtered, len(pid2n)*len(uid2n), lf*100))

    return like_matrix, pages, hoaxes
