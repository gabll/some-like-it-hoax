import os
from data_loader import *
from post_getter import *


SINCE = '2016-07-01'
TO = '2016-12-31'
FILE_PAGES = 'pages_to_read.txt'
JSON_FILENAME = 'tacchiniEtAl_fbposts_2016S2.json'


def downloadPostFromPages():
    keys_to_delete = ['likes', 'comments', 'privacy', 'actions', 'icon']
    print("Get all Facebook posts since "+SINCE+" to "+TO, file=sys.stderr)
    pageid_list = getAllPagesIds()
    for pageid in pageid_list:
        print(pageid + " ... ", end="", flush=True, file=sys.stderr)
        outfilename = pageid + '_' + SINCE + '_' + TO + '.json'
        if os.path.exists(outfilename):
            print(" already saved.", flush=True, file=sys.stderr)
            continue
        posts = list()
        for post in gen_all_posts(pageid, SINCE, TO,
                                  post_filter=lambda x: 'message' in x.keys(),
                                  keys_to_delete=keys_to_delete):
            posts.append(post)
        save_json(posts, outfilename)
        print("saved.", file=sys.stderr)


def getAllPagesIds():
    res = list()
    with open(FILE_PAGES) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 2 and '#' is not line[0]:
                pageid = line.split('#')[0].strip()
                res.append(pageid)
    return res


def joinFilesOfPages():
    pageids = getAllPagesIds()
    jsonfiles = [x + '_'+SINCE+'_'+TO+'.json' for x in pageids]
    posts = load_json(jsonfiles)
    save_json(posts, JSON_FILENAME)


def addLikers(posts: list):
    SAVE_EVERY = 50
    posts_len = len(posts)
    error_count = 0
    changes = False
    for i, post in enumerate(posts):
        if 'likes' not in post:
            add_likers(post)
            changes = True
        sys.stdout.write("\r%d/%d" % (i + 1, posts_len))
        sys.stdout.flush()
        if (i + 1) % SAVE_EVERY == 0 and changes:
            save_json(posts, JSON_FILENAME)
            #print(" - Saved.")

    save_json(posts, JSON_FILENAME)
    print("%d errors on %d downloads" % (error_count, posts_len))


def saveLikers(posts):
    posts_dict = dict()
    for post in posts:
        if 'likes' in post and len(post['likes']) > 0:
            posts_dict[post['id']] = post['likes']
    save_json(posts_dict, 'likers.json')


if __name__ == '__main__':
    """ STEP 1: Download all posts from page list """
    downloadPostFromPages()
    joinFilesOfPages()

    """ STEP 2: Add all likes"""
    posts = load_json(JSON_FILENAME)
    addLikers(posts)

    """ STEP 3: Save likers format """
    saveLikers(posts)