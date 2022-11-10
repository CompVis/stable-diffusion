import threading
import requests
import json
import random
from pybooru import Danbooru
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--danbooru_username', '-user', type=str, required=False)
parser.add_argument('--danbooru_key', '-key', type=str, required=False)
parser.add_argument('--tags', '-t', required=False, default="solo -comic -animated -touhou -rating:general order:score age:<1month")
parser.add_argument('--posts', '-p', required=False, type=int, default=10000)
parser.add_argument('--output', '-o', required=False, default='links.json')
parser.add_argument('--start_page', '-s', required=False, default=0, type=int)
args = parser.parse_args()

import re

def clean(text: str):
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.split(' ')
    new_text = []
    for i in text:
        new_text.append(i.lstrip('_').rstrip('_'))
    text = set(new_text)
    text = ' '.join(text)
    text = text.lstrip().rstrip()
    return text

def set_val(val_dict, new_dict, key, clean_val = True):
    if (key in val_dict) and val_dict[key]:
        if clean_val:
            new_dict[key] = clean(val_dict[key])
        else:
            new_dict[key] = val_dict[key]
        return new_dict

class DanbooruScraper():
    def __init__(self, username, key):
        self.username = username
        self.key = key
        self.dbclient = Danbooru('danbooru', username=self.username, api_key=self.key)

    # This will get danbooru urls and tags, put them in a dict, then write as a json file
    def get_urls(self, tags, num_posts, batch_size, file="data_urls.json"):
        dict = {}
        if num_posts % batch_size != 0:
            print("Error: num_posts must be divisible by batch_size")
            return
        for i in tqdm(range(num_posts//batch_size)):
            try:
                urls = self.dbclient.post_list(tags=tags, limit=batch_size, random=False, page=i+args.start_page)
            except Exception as e:
                print(f'Skipping page {i} - {e}')
                continue
            if not urls:
                print(f'Empty results at {i}')
                break
            for j in urls:
                if 'file_url' in j:
                    if j['file_url'] not in dict:
                        d_tags = {}
                        if ('tag_string_copyright' in j) and j['tag_string_copyright']:
                            d_tags = set_val(j, d_tags, 'tag_string_copyright')
                        if ('tag_string_artist' in j) and j['tag_string_artist']:
                            d_tags = set_val(j, d_tags, 'tag_string_artist')
                        if ('tag_string_character' in j) and j['tag_string_character']:
                            d_tags = set_val(j, d_tags, 'tag_string_character')
                        if ('tag_string_general' in j) and j['tag_string_general']:
                            d_tags = set_val(j, d_tags, 'tag_string_general')           
                        if ('tag_string_meta' in j) and j['tag_string_meta']:
                            d_tags = set_val(j, d_tags, 'tag_string_meta')
                        d_tags['file_url'] = j['file_url']
                        dict[j['id']] = d_tags
                else:
                    print("Error: file_url not found")
        with open(file, 'w') as f:
            json.dump(dict, f)

# now test
if __name__ == "__main__":
    ds = DanbooruScraper(args.danbooru_username, args.danbooru_key)
    ds.get_urls(args.tags, args.posts, 100, file=args.output)
