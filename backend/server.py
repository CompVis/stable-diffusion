import mimetypes
import transformers
import json
import os
import traceback
import eventlet
import glob
import shlex
import argparse

from flask_socketio import SocketIO
from flask import Flask, send_from_directory, url_for, jsonify
from pathlib import Path
from PIL import Image
from pytorch_lightning import logging
from threading import Event
from uuid import uuid4

from ldm.gfpgan.gfpgan_tools import real_esrgan_upscale
from ldm.gfpgan.gfpgan_tools import run_gfpgan
from ldm.generate import Generate
from ldm.dream.pngwriter import PngWriter, retrieve_metadata

from modules.parameters import parameters_to_command, create_cmd_parser


"""
USER CONFIG
"""

output_dir = "outputs/"  # Base output directory for images
#host = 'localhost'  # Web & socket.io host
host = '0.0.0.0'  # Web & socket.io host
port = 9090  # Web & socket.io port
verbose = False # enables copious socket.io logging
additional_allowed_origins = ['http://localhost:9090'] # additional CORS allowed origins


"""
END USER CONFIG
"""


"""
SERVER SETUP
"""


# fix missing mimetypes on windows due to registry wonkiness
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

app = Flask(__name__, static_url_path='', static_folder='../frontend/dist/')


app.config['OUTPUTS_FOLDER'] = "../outputs"


@app.route('/outputs/<path:filename>')
def outputs(filename):
    return send_from_directory(
        app.config['OUTPUTS_FOLDER'],
        filename
    )


@app.route("/", defaults={'path': ''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')


logger = True if verbose else False
engineio_logger = True if verbose else False

# default 1,000,000, needs to be higher for socketio to accept larger images
max_http_buffer_size = 10000000

cors_allowed_origins = [f"http://{host}:{port}"] + additional_allowed_origins

socketio = SocketIO(
                        app,
                        logger=logger,
                        engineio_logger=engineio_logger,
                        max_http_buffer_size=max_http_buffer_size,
                        cors_allowed_origins=cors_allowed_origins,
                    )


"""
END SERVER SETUP
"""


"""
APP SETUP
"""


class CanceledException(Exception):
    pass


canceled = Event()

# reduce logging outputs to error
transformers.logging.set_verbosity_error()
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# Initialize and load model
model = Generate()
model.load_model()


# location for "finished" images
result_path = os.path.join(output_dir, 'img-samples/')

# temporary path for intermediates
intermediate_path = os.path.join(result_path, 'intermediates/')

# path for user-uploaded init images and masks
init_path = os.path.join(result_path, 'init-images/')
mask_path = os.path.join(result_path, 'mask-images/')

# txt log
log_path = os.path.join(result_path, 'dream_log.txt')

# make all output paths
[os.makedirs(path, exist_ok=True)
 for path in [result_path, intermediate_path, init_path, mask_path]]


"""
END APP SETUP
"""


"""
SOCKET.IO LISTENERS
"""


@socketio.on('requestAllImages')
def handle_request_all_images():
    print(f'>> All images requested')
    parser = create_cmd_parser()
    paths = list(filter(os.path.isfile, glob.glob(result_path + "*.png")))
    paths.sort(key=lambda x: os.path.getmtime(x))
    image_array = []
    for path in paths:
        # image = Image.open(path)
        all_metadata = retrieve_metadata(path)
        if 'Dream' in all_metadata and not all_metadata['sd-metadata']:
            metadata = vars(parser.parse_args(shlex.split(all_metadata['Dream'])))
        else:
            metadata = all_metadata['sd-metadata']
        image_array.append({'path': path, 'metadata': metadata})
    return make_response("OK", data=image_array)


@socketio.on('generateImage')
def handle_generate_image_event(generation_parameters, esrgan_parameters, gfpgan_parameters):
    print(f'>> Image generation requested: {generation_parameters}\nESRGAN parameters: {esrgan_parameters}\nGFPGAN parameters: {gfpgan_parameters}')
    generate_images(
        generation_parameters,
        esrgan_parameters,
        gfpgan_parameters
    )
    return make_response("OK")


@socketio.on('runESRGAN')
def handle_run_esrgan_event(original_image, esrgan_parameters):
    print(f'>> ESRGAN upscale requested for "{original_image["url"]}": {esrgan_parameters}')
    image = Image.open(original_image["url"])

    seed = original_image['metadata']['seed'] if 'seed' in original_image['metadata'] else 'unknown_seed'

    image = real_esrgan_upscale(
        image=image,
        upsampler_scale=esrgan_parameters['upscale'][0],
        strength=esrgan_parameters['upscale'][1],
        seed=seed
    )

    esrgan_parameters['seed'] = seed
    path = save_image(image, esrgan_parameters, result_path, postprocessing='esrgan')
    command = parameters_to_command(esrgan_parameters)

    write_log_message(f'[Upscaled] "{original_image["url"]}" > "{path}": {command}')

    socketio.emit(
            'result', {'url': os.path.relpath(path), 'type': 'esrgan', 'uuid': original_image['uuid'],'metadata': esrgan_parameters})



@socketio.on('runGFPGAN')
def handle_run_gfpgan_event(original_image, gfpgan_parameters):
    print(f'>> GFPGAN face fix requested for "{original_image["url"]}": {gfpgan_parameters}')
    image = Image.open(original_image["url"])

    seed = original_image['metadata']['seed'] if 'seed' in original_image['metadata'] else 'unknown_seed'

    image = run_gfpgan(
        image=image,
        strength=gfpgan_parameters['gfpgan_strength'],
        seed=seed,
        upsampler_scale=1
    )

    gfpgan_parameters['seed'] = seed
    path = save_image(image, gfpgan_parameters, result_path, postprocessing='gfpgan')
    command = parameters_to_command(gfpgan_parameters)

    write_log_message(f'[Fixed faces] "{original_image["url"]}" > "{path}": {command}')

    socketio.emit(
            'result', {'url': os.path.relpath(path), 'type': 'gfpgan', 'uuid': original_image['uuid'],'metadata': gfpgan_parameters})


@socketio.on('cancel')
def handle_cancel():
    print(f'>> Cancel processing requested')
    canceled.set()
    return make_response("OK")


# TODO: I think this needs a safety mechanism.
@socketio.on('deleteImage')
def handle_delete_image(path):
    print(f'>> Delete requested "{path}"')
    Path(path).unlink()
    return make_response("OK")


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadInitialImage')
def handle_upload_initial_image(bytes, name):
    print(f'>> Init image upload requested "{name}"')
    uuid = uuid4().hex
    split = os.path.splitext(name)
    name = f'{split[0]}.{uuid}{split[1]}'
    file_path = os.path.join(init_path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    newFile = open(file_path, "wb")
    newFile.write(bytes)
    return make_response("OK", data=file_path)


# TODO: I think this needs a safety mechanism.
@socketio.on('uploadMaskImage')
def handle_upload_mask_image(bytes, name):
    print(f'>> Mask image upload requested "{name}"')
    uuid = uuid4().hex
    split = os.path.splitext(name)
    name = f'{split[0]}.{uuid}{split[1]}'
    file_path = os.path.join(mask_path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    newFile = open(file_path, "wb")
    newFile.write(bytes)
    return make_response("OK", data=file_path)



"""
END SOCKET.IO LISTENERS
"""



"""
ADDITIONAL FUNCTIONS
"""


def write_log_message(message, log_path=log_path):
    """Logs the filename and parameters used to generate or process that image to log file"""
    message = f'{message}\n'
    with open(log_path, 'a', encoding='utf-8') as file:
        file.writelines(message)


def make_response(status, message=None, data=None):
    response = {'status': status}
    if message is not None:
        response['message'] = message
    if data is not None:
        response['data'] = data
    return response


def save_image(image, parameters, output_dir, step_index=None, postprocessing=False):
    seed = parameters['seed'] if 'seed' in parameters else 'unknown_seed'

    pngwriter = PngWriter(output_dir)
    prefix = pngwriter.unique_prefix()

    filename = f'{prefix}.{seed}'

    if step_index:
        filename += f'.{step_index}'
    if postprocessing:
        filename += f'.postprocessed'

    filename += '.png'

    command = parameters_to_command(parameters)

    path = pngwriter.save_image_and_prompt_to_png(image, command, metadata=parameters, name=filename)

    return path

def generate_images(generation_parameters, esrgan_parameters, gfpgan_parameters):
    canceled.clear()

    step_index = 1

    def image_progress(sample, step):
        if canceled.is_set():
            raise CanceledException
        nonlocal step_index
        nonlocal generation_parameters
        if generation_parameters["progress_images"] and step % 5 == 0 and step < generation_parameters['steps'] - 1:
            image = model.sample_to_image(sample)
            path = save_image(image, generation_parameters, intermediate_path, step_index)

            step_index += 1
            socketio.emit('intermediateResult', {
                          'url': os.path.relpath(path), 'metadata': generation_parameters})
        socketio.emit('progress', {'step': step + 1})
        eventlet.sleep(0)

    def image_done(image, seed):
        nonlocal generation_parameters
        nonlocal esrgan_parameters
        nonlocal gfpgan_parameters

        all_parameters = generation_parameters
        postprocessing = False

        if esrgan_parameters:
            image = real_esrgan_upscale(
                image=image,
                strength=esrgan_parameters['strength'],
                upsampler_scale=esrgan_parameters['level'],
                seed=seed
            )
            postprocessing = True
            all_parameters["upscale"] = [esrgan_parameters['level'], esrgan_parameters['strength']]

        if gfpgan_parameters:
            image = run_gfpgan(
                image=image,
                strength=gfpgan_parameters['strength'],
                seed=seed,
                upsampler_scale=1,
            )
            postprocessing = True
            all_parameters["gfpgan_strength"] = gfpgan_parameters['strength']

        all_parameters['seed'] = seed

        path = save_image(image, all_parameters, result_path, postprocessing=postprocessing)
        command = parameters_to_command(all_parameters)

        print(f'Image generated: "{path}"')
        write_log_message(f'[Generated] "{path}": {command}')

        socketio.emit(
            'result', {'url': os.path.relpath(path), 'type': 'generation', 'metadata': all_parameters})
        eventlet.sleep(0)

    try:
        model.prompt2image(
            **generation_parameters,
            step_callback=image_progress,
            image_callback=image_done
        )

    except KeyboardInterrupt:
        raise
    except CanceledException:
        pass
    except Exception as e:
        socketio.emit('error', (str(e)))
        print("\n")
        traceback.print_exc()
        print("\n")


"""
END ADDITIONAL FUNCTIONS
"""


if __name__ == '__main__':
    print(f'Starting server at http://{host}:{port}')
    socketio.run(app, host=host, port=port)
