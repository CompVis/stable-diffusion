import os
from flask import Flask, request, jsonify
from txt2img import generate as generateImage
from dotenv import load_dotenv

load_dotenv('../.env')

app = Flask(__name__)

def verifyApiKey():
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

    if prompt is not None and verifyApiKey():
        results = generateImage(
            prompt = prompt,
            seed = args.get('seed', 0),
            width = args.get('width', 512),
            height = args.get('height', 512),
            steps = args.get('steps', 40),
            iterations = args.get('iterations', 1),
            scale = args.get('scale', 7.5),
        )

        return jsonify(
            data=results
        )
    else:
        return jsonify({'error': 'No prompt provided.'}), 406


@app.route('/', methods=['GET'])
def heartbeat():
    return 'OK!'

app.run(host='0.0.0.0', port=3001, debug=True)