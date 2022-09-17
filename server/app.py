import os
import redis
import json
import uuid
from os.path import join, dirname

from flask import Flask, request, jsonify
from time import time
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '../.env')
load_dotenv(dotenv_path)

app = Flask(__name__)

r = redis.Redis(
    host= os.environ.get('REDIS_HOST', 'redis://localhost'),
    port= os.environ.get('REDIS_PORT', '6379'),
    password= os.environ.get('REDIS_PASSWORD'),
    ssl=False
)


def verify_api_key():
    """Verify that the API key is valid"""
    if request.headers.get('X-API-KEY') == os.environ.get('API_KEY'):
        return True
    else:
        return False


@app.route('/', methods=['POST'])
def generate():
    """Generate an image from a prompt"""
    args = request.json
    prompt = args.get('prompt', None)
    webhook_url = request.headers.get('X-WEBHOOK-URL', None)
    job_id = uuid.uuid4().hex

    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    if not webhook_url:
        return jsonify({'error': 'No webhook URL provided'}), 400

    if not verify_api_key():
        return jsonify({'error': 'Invalid API key'}), 401

    params = {
        'job_id': job_id,
        'webhook_url': webhook_url,
        'seed': args.get('seed', 0),
        'prompt': args.get('prompt', None),
        'width': args.get('width', 512),
        'height': args.get('height', 512),
        'steps': args.get('steps', 40),
        'iterations': args.get('iterations', 1),
        'scale': args.get('scale', 7.5),
    }

    r.rpush('generate_images', json.dumps(obj=params))
    
    return jsonify({
        'job_id': job_id,
    })


@app.route('/webhook', methods=['POST'])
def test_webhook():
    print(request.json)
    return 'OK!'


@app.route('/', methods=['GET'])
def heartbeat():
    return 'OK!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=False)