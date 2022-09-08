import redis
import time
import json
from txt2img import generate as generate_image
from dotenv import load_dotenv
import os
from os.path import join, dirname
from dotenv import load_dotenv
import requests

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

r = redis.Redis()

def create_image(prompt, args):
    """Create image and upload to S3"""
    webhook = args.get('webhook_url')

    if not webhook:
        return
        
    try:
        images, time = generate_image(
            prompt = prompt,
            seed = args.get('seed', 0),
            width = args.get('width', 512),
            height = args.get('height', 512),
            steps = args.get('steps', 40),
            iterations = args.get('iterations', 1),
            scale = args.get('scale', 7.5),
        )

        requests.post(webhook, json={ 'images': images, 'time': time, 'job_id': args.get('job_id') })
    except Exception as e:
        print(e)


while True:
    data = r.lpop('generate_images')

    if data:
        data = json.loads(data)
        create_image(data['prompt'], data)

