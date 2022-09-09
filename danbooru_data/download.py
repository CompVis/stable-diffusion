import os
import json
import requests
import multiprocessing
import tqdm

# downloads URLs from JSON

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, required=False)
parser.add_argument('--out_dir', '-o', type=str, required=False)
parser.add_argument('--threads', '-p', required=False, default=32)
args = parser.parse_args()

class DownloadManager():
    def __init__(self, max_threads=32):
        self.failed_downloads = []
        self.max_threads = max_threads
    
    # args = (link, metadata, out_img_dir, out_text_dir)
    def download(self, args):
        try:
            r = requests.get(args[0], stream=True)
            with open(args[2] + args[0].split('/')[-1], 'wb') as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            with open(args[3] + args[0].split('/')[-1].split('.')[0] + '.txt', 'w') as f:
                f.write(args[1])
        except:
            self.failed_downloads.append((args[0], args[1]))
    
    def download_urls(self, file_path, out_dir):
        with open(file_path) as f:
            data = json.load(f)
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            os.makedirs(out_dir + '/img')
            os.makedirs(out_dir + '/text')
        
        thread_args = []

        print(f'Loading {file_path} for download on {self.max_threads} threads...')

        # create initial thread_args
        for k, v in tqdm.tqdm(data.items()):
            thread_args.append((k, v, out_dir + 'img/', out_dir + 'text/'))
        
        # divide thread_args into chunks divisible by max_threads
        chunks = []
        for i in range(0, len(thread_args), self.max_threads):
            chunks.append(thread_args[i:i+self.max_threads])
        
        print(f'Downloading {len(thread_args)} images...')

        # download chunks synchronously
        for chunk in tqdm.tqdm(chunks):
            with multiprocessing.Pool(self.max_threads) as p:
                p.map(self.download, chunk)

        if len(self.failed_downloads) > 0:
            print("Failed downloads:")
            for i in self.failed_downloads:
                print(i[0])
            print("\n")
        """        
        # attempt to download any remaining failed downloads
        print('\nAttempting to download any failed downloads...')
        print('Failed downloads:', len(self.failed_downloads))
        if len(self.failed_downloads) > 0:
            for url in tqdm.tqdm(self.failed_downloads):
                self.download((url[0], url[1], out_dir + 'img/', out_dir + 'text/'))
        """
    
        
if __name__ == '__main__':
    dm = DownloadManager(max_threads=args.threads)
    dm.download_urls(args.file, args.out_dir)