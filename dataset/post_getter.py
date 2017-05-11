#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import facebook
import requests
from datetime import datetime
import dateutil.parser
import json
import sys
import credentials


_graph = None


def init() -> None:
    """
    Init the facebook connection with the access_token stored in credentials
    :return: None
    """
    global _graph
    _graph = facebook.GraphAPI(access_token=credentials.FACEBOOK_ACCESS_TOKEN)


def gen_all_posts(pageid: str, date_from: str, date_to: str,
                  post_filter: object = lambda x: True,
                  keys_to_delete: list = []) \
        -> list:
    """
    Generator of all post of a page since a date (most recent first).
    :param pageid:
    :param date_from: date as string
    :param date_to:
    :param post_filter:
    :param keys_to_delete:
    :return: posts (as generator)
    """
    # Posts are returned by the GraphAPI in paging mode
    if _graph is None:
        init()
    page_posts = _graph.get_object(id=pageid + '/posts')
    stop_date = dateutil.parser.parse(date_from)
    begin_date = dateutil.parser.parse(date_to)
    while True:
        for post in page_posts['data']:
            post_date = dateutil.parser.parse(post['created_time'])
            # Remove tzinfo for the comparison with stop_date
            post_date = datetime.replace(post_date, tzinfo=None)
            if post_date >= stop_date:
                if post_date < begin_date and post_filter(post):
                    yield distill_post(post, keys_to_delete)
            else:
                # No more posts to generate
                return
        if 'next' in page_posts['paging']:
            page_posts = requests.get(page_posts['paging']['next']).json()
        else:
            break


def gen_likers(postid):
    if _graph is None:
        init()
    res = _graph.get_object(id=postid + '/likes')
    while True:
        if 'data' not in res:
            break
        for elem in res['data']:
            yield elem['id']
        if len(res['data']) > 0 and 'next' in res['paging']:
            res = requests.get(res['paging']['next']).json()
        else:
            break


def add_likers(post):
    try:
        likes = list(gen_likers(post['id']))
        post['likes'] = likes
    except facebook.GraphAPIError:
        print("Warning: post " + post['id'] + " not found. Skipped.")


def distill_post(post, keys_to_delete):
    """
    Remove unwanted keys from post dictionary
    :param post:
    :param keys_to_delete:
    :return:
    """
    for key_to_delete in keys_to_delete:
        if key_to_delete in post.keys():
            del(post[key_to_delete])
    return post
