#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#

import json
import requests
from bs4 import BeautifulSoup


def request_url_http(url):
    """HTTP request to get the HTML of the specified URL"""
    r = requests.get(url)
    r.encoding = 'utf-8'
    html = r.text
    return html


def get_soup(url):
    """Create a soup from URL"""
    html = request_url_http(url)
    soup = BeautifulSoup(html, "lxml")
    return soup


def load_json(files):
    if type(files) is not list:
        files = [files]

    joined_json = list()
    for filename in files:
        with open(filename) as f:
            _data = json.load(f)
            joined_json.extend(_data)
    return joined_json


def save_json(datastruct, filename):
    with open(filename, 'w') as f:
        json.dump(datastruct, f, indent=2, ensure_ascii=False)

