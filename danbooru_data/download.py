import os
import json
import requests
import multiprocessing
import tqdm
import webdataset
from concurrent import futures
import io
import tarfile
import glob
import uuid

from PIL import Image

# downloads URLs from JSON

import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, required=False, default='links.json')
parser.add_argument('--out_file', '-o', type=str, required=False, default='dataset-%06d.tar')
parser.add_argument('--max_size', '-m', type=int, required=False, default=4294967296)
parser.add_argument('--threads', '-p', required=False, default=16)
args = parser.parse_args()

class DownloadManager():
    def __init__(self, max_threads: int = 32):
        self.failed_downloads = []
        self.max_threads = max_threads
        self.uuid = str(uuid.uuid1())
    
    # args = (post_id, link, caption_data)
    def download(self, args):
        try:
            image = Image.open(requests.get(args[1], stream=True).raw).convert('RGB')
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='PNG')
            __key__ = '%07d' % int(args[0])
            image = image_bytes.getvalue()
            caption = str(json.dumps(args[2]))

            with open(f'{self.uuid}/{__key__}.image', 'wb') as f:
                f.write(image)
            with open(f'{self.uuid}/{__key__}.caption', 'w') as f:
                f.write(caption)

        except Exception as e:
            print(e)
            self.failed_downloads.append((args[0], args[1], args[2]))
    
    def download_urls(self, file_path):
        with open(file_path) as f:
            data = json.load(f)
        thread_args = []

        delimiter = '\\' if os.name == 'nt' else '/'

        self.uuid = (file_path.split(delimiter)[-1]).split('.')[0]
        
        if not os.path.exists(f'./{self.uuid}'):
            os.mkdir(f'{self.uuid}')

        print(f'Loading {file_path} for downloading on {self.max_threads} threads... Writing to dataset {self.uuid}')

        # create initial thread_args
        for k, v in tqdm.tqdm(data.items()):
            thread_args.append((k, v['file_url'], v))
        
        # divide thread_args into chunks divisible by max_threads
        chunks = []
        for i in range(0, len(thread_args), self.max_threads):
            chunks.append(thread_args[i:i+self.max_threads])
        
        print(f'Downloading {len(thread_args)} images...')

        # download chunks synchronously
        for chunk in tqdm.tqdm(chunks):
            with futures.ThreadPoolExecutor(args.threads) as p:
                p.map(self.download, chunk)

        if len(self.failed_downloads) > 0:
            print("Failed downloads:")
            for i in self.failed_downloads:
                print(i[0])
            print("\n")
        
        # put things into tar
        print(f'Writing webdataset to {self.uuid}')
        archive = tarfile.open(f'{self.uuid}.tar', 'w')
        files = glob.glob(f'{self.uuid}/*')
        for f in tqdm.tqdm(files):
            archive.add(f, f.split(delimiter)[-1])
        
        archive.close()
        
        print('Cleaning up...')
        shutil.rmtree(self.uuid)
        
if __name__ == '__main__':
    dm = DownloadManager(max_threads=args.threads)
    dm.download_urls(args.file)