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
parser.add_argument('--posts', '-p', required=False, default=10000)
parser.add_argument('--output', '-o', required=False, default='links.json')
args = parser.parse_args()

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
            urls = self.dbclient.post_list(tags=tags, limit=batch_size, random=False, page=i)
            if not urls:
                print(f'Empty results at {i}')
                break
            for j in urls:
                if 'file_url' in j:
                    if j['file_url'] not in dict:
                        d_url = j['file_url']
                        d_tags = j['tag_string_copyright'] + " " + j['tag_string_character'] + " " + j['tag_string_general'] + " " + j['tag_string_artist']

                        dict[d_url] = d_tags
                else:
                    print("Error: file_url not found")
        with open(file, 'w') as f:
            json.dump(dict, f)

# now test
if __name__ == "__main__":
    ds = DanbooruScraper(args.danbooru_username, args.danbooru_key)
    ds.get_urls(args.tags, args.posts, 100, file=args.output)