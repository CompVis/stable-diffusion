import mimetypes
import transformers
import json
import os
import traceback
import eventlet
import glob
import shlex
import math
import shutil
import sys

sys.path.append(".")

from argparse import ArgumentTypeError
from modules.create_cmd_parser import create_cmd_parser

parser = create_cmd_parser()
opt = parser.parse_args()


from flask_socketio import SocketIO
from flask import Flask, send_from_directory, url_for, jsonify
from pathlib import Path
from PIL import Image
from pytorch_lightning import logging
from threading import Event
from uuid import uuid4
from send2trash import send2trash


from ldm.generate import Generate
from ldm.invoke.restoration import Restoration
from ldm.invoke.pngwriter import PngWriter, retrieve_metadata
from ldm.invoke.args import APP_ID, APP_VERSION, calculate_init_img_hash
from ldm.invoke.conditioning import split_weighted_subprompts

from modules.parameters import parameters_to_command


"""
USER CONFIG
"""
if opt.cors and "*" in opt.cors:
    raise ArgumentTypeError('"*" is not an allowed CORS origin')


output_dir = "outputs/"  # Base output directory for images
host = opt.host  # Web & socket.io host
port = opt.port  # Web & socket.io port
verbose = opt.verbose  # enables copious socket.io logging
precision = opt.precision
free_gpu_mem = opt.free_gpu_mem
embedding_path = opt.embedding_path
additional_allowed_origins = (
    opt.cors if opt.cors else []
)  # additional CORS allowed origins
model = "stable-diffusion-1.4"

"""
END USER CONFIG
"""


print("* Initializing, be patient...\n")


"""
SERVER SETUP
"""


# fix missing mimetypes on windows due to registry wonkiness
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

app = Flask(__name__, static_url_path="", static_folder="../frontend/dist/")


app.config["OUTPUTS_FOLDER"] = "../outputs"


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(app.config["OUTPUTS_FOLDER"], filename)


@app.route("/", defaults={"path": ""})
def serve(path):
    return send_from_directory(app.static_folder, "index.html")


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
    ping_interval=(50, 50),
    ping_timeout=60,
)


"""
END SERVER SETUP
"""


"""
APP SETUP
"""


class CanceledException(Exception):
    pass


try:
    gfpgan, codeformer, esrgan = None, None, None
    from ldm.invoke.restoration.base import Restoration

    restoration = Restoration()
    gfpgan, codeformer = restoration.load_face_restore_models()
    esrgan = restoration.load_esrgan()

    # coreformer.process(self, image, strength, device, seed=None, fidelity=0.75)

except (ModuleNotFoundError, ImportError):
    print(traceback.format_exc(), file=sys.stderr)
    print(">> You may need to install the ESRGAN and/or GFPGAN modules")

canceled = Event()

# reduce logging outputs to error
transformers.logging.set_verbosity_error()
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Initialize and load model
generate = Generate(
    model,
    precision=precision,
    embedding_path=embedding_path,
)
generate.free_gpu_mem = free_gpu_mem
generate.load_model()


# location for "finished" images
result_path = os.path.join(output_dir, "img-samples/")

# temporary path for intermediates
intermediate_path = os.path.join(result_path, "intermediates/")

# path for user-uploaded init images and masks
init_image_path = os.path.join(result_path, "init-images/")
mask_image_path = os.path.join(result_path, "mask-images/")

# txt log
log_path = os.path.join(result_path, "invoke_log.txt")

# make all output paths
[
    os.makedirs(path, exist_ok=True)
    for path in [result_path, intermediate_path, init_image_path, mask_image_path]
]


"""
END APP SETUP
"""


"""
SOCKET.IO LISTENERS
"""


@socketio.on("requestSystemConfig")
def handle_request_capabilities():
    print(f">> System config requested")
    config = get_system_config()
    socketio.emit("systemConfig", config)


@socketio.on("requestImages")
def handle_request_images(page=1, offset=0, last_mtime=None):
    chunk_size = 50

    if last_mtime:
        print(f">> Latest images requested")
    else:
        print(
            f">> Page {page} of images requested (page size {chunk_size} offset {offset})"
        )

    paths = glob.glob(os.path.join(result_path, "*.png"))
    sorted_paths = sorted(paths, key=lambda x: os.path.getmtime(x), reverse=True)

    if last_mtime:
        image_paths = filter(lambda x: os.path.getmtime(x) > last_mtime, sorted_paths)
    else:

        image_paths = sorted_paths[
            slice(chunk_size * (page - 1) + offset, chunk_size * page + offset)
        ]
        page = page + 1

    image_array = []

    for path in image_paths:
        metadata = retrieve_metadata(path)
        image_array.append(
            {
                "url": path,
                "mtime": os.path.getmtime(path),
                "metadata": metadata["sd-metadata"],
            }
        )

    socketio.emit(
        "galleryImages",
        {
            "images": image_array,
            "nextPage": page,
            "offset": offset,
            "onlyNewImages": True if last_mtime else False,
        },
    )


@socketio.on("generateImage")
def handle_generate_image_event(
    generation_parameters, esrgan_parameters, gfpgan_parameters
):
    print(
        f">> Image generation requested: {generation_parameters}\nESRGAN parameters: {esrgan_parameters}\nGFPGAN parameters: {gfpgan_parameters}"
    )
    generate_images(generation_parameters, esrgan_parameters, gfpgan_parameters)


@socketio.on("runESRGAN")
def handle_run_esrgan_event(original_image, esrgan_parameters):
    print(
        f'>> ESRGAN upscale requested for "{original_image["url"]}": {esrgan_parameters}'
    )
    progress = {
        "currentStep": 1,
        "totalSteps": 1,
        "currentIteration": 1,
        "totalIterations": 1,
        "currentStatus": "Preparing",
        "isProcessing": True,
        "currentStatusHasSteps": False,
    }

    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    image = Image.open(original_image["url"])

    seed = (
        original_image["metadata"]["seed"]
        if "seed" in original_image["metadata"]
        else "unknown_seed"
    )

    progress["currentStatus"] = "Upscaling"
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    image = esrgan.process(
        image=image,
        upsampler_scale=esrgan_parameters["upscale"][0],
        strength=esrgan_parameters["upscale"][1],
        seed=seed,
    )

    progress["currentStatus"] = "Saving image"
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    esrgan_parameters["seed"] = seed
    metadata = parameters_to_post_processed_image_metadata(
        parameters=esrgan_parameters,
        original_image_path=original_image["url"],
        type="esrgan",
    )
    command = parameters_to_command(esrgan_parameters)

    path = save_image(image, command, metadata, result_path, postprocessing="esrgan")

    write_log_message(f'[Upscaled] "{original_image["url"]}" > "{path}": {command}')

    progress["currentStatus"] = "Finished"
    progress["currentStep"] = 0
    progress["totalSteps"] = 0
    progress["currentIteration"] = 0
    progress["totalIterations"] = 0
    progress["isProcessing"] = False
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    socketio.emit(
        "esrganResult",
        {
            "url": os.path.relpath(path),
            "mtime": os.path.getmtime(path),
            "metadata": metadata,
        },
    )


@socketio.on("runGFPGAN")
def handle_run_gfpgan_event(original_image, gfpgan_parameters):
    print(
        f'>> GFPGAN face fix requested for "{original_image["url"]}": {gfpgan_parameters}'
    )
    progress = {
        "currentStep": 1,
        "totalSteps": 1,
        "currentIteration": 1,
        "totalIterations": 1,
        "currentStatus": "Preparing",
        "isProcessing": True,
        "currentStatusHasSteps": False,
    }

    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    image = Image.open(original_image["url"])

    seed = (
        original_image["metadata"]["seed"]
        if "seed" in original_image["metadata"]
        else "unknown_seed"
    )

    progress["currentStatus"] = "Fixing faces"
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    image = gfpgan.process(
        image=image, strength=gfpgan_parameters["gfpgan_strength"], seed=seed
    )

    progress["currentStatus"] = "Saving image"
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    gfpgan_parameters["seed"] = seed
    metadata = parameters_to_post_processed_image_metadata(
        parameters=gfpgan_parameters,
        original_image_path=original_image["url"],
        type="gfpgan",
    )
    command = parameters_to_command(gfpgan_parameters)

    path = save_image(image, command, metadata, result_path, postprocessing="gfpgan")

    write_log_message(f'[Fixed faces] "{original_image["url"]}" > "{path}": {command}')

    progress["currentStatus"] = "Finished"
    progress["currentStep"] = 0
    progress["totalSteps"] = 0
    progress["currentIteration"] = 0
    progress["totalIterations"] = 0
    progress["isProcessing"] = False
    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    socketio.emit(
        "gfpganResult",
        {
            "url": os.path.relpath(path),
            "mtime": os.path.mtime(path),
            "metadata": metadata,
        },
    )


@socketio.on("cancel")
def handle_cancel():
    print(f">> Cancel processing requested")
    canceled.set()
    socketio.emit("processingCanceled")


# TODO: I think this needs a safety mechanism.
@socketio.on("deleteImage")
def handle_delete_image(path, uuid):
    print(f'>> Delete requested "{path}"')
    send2trash(path)
    socketio.emit("imageDeleted", {"url": path, "uuid": uuid})


# TODO: I think this needs a safety mechanism.
@socketio.on("uploadInitialImage")
def handle_upload_initial_image(bytes, name):
    print(f'>> Init image upload requested "{name}"')
    uuid = uuid4().hex
    split = os.path.splitext(name)
    name = f"{split[0]}.{uuid}{split[1]}"
    file_path = os.path.join(init_image_path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    newFile = open(file_path, "wb")
    newFile.write(bytes)
    socketio.emit("initialImageUploaded", {"url": file_path, "uuid": ""})


# TODO: I think this needs a safety mechanism.
@socketio.on("uploadMaskImage")
def handle_upload_mask_image(bytes, name):
    print(f'>> Mask image upload requested "{name}"')
    uuid = uuid4().hex
    split = os.path.splitext(name)
    name = f"{split[0]}.{uuid}{split[1]}"
    file_path = os.path.join(mask_image_path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    newFile = open(file_path, "wb")
    newFile.write(bytes)
    socketio.emit("maskImageUploaded", {"url": file_path, "uuid": ""})


"""
END SOCKET.IO LISTENERS
"""


"""
ADDITIONAL FUNCTIONS
"""


def get_system_config():
    return {
        "model": "stable diffusion",
        "model_id": model,
        "model_hash": generate.model_hash,
        "app_id": APP_ID,
        "app_version": APP_VERSION,
    }


def parameters_to_post_processed_image_metadata(parameters, original_image_path, type):
    # top-level metadata minus `image` or `images`
    metadata = get_system_config()

    orig_hash = calculate_init_img_hash(original_image_path)

    image = {"orig_path": original_image_path, "orig_hash": orig_hash}

    if type == "esrgan":
        image["type"] = "esrgan"
        image["scale"] = parameters["upscale"][0]
        image["strength"] = parameters["upscale"][1]
    elif type == "gfpgan":
        image["type"] = "gfpgan"
        image["strength"] = parameters["gfpgan_strength"]
    else:
        raise TypeError(f"Invalid type: {type}")

    metadata["image"] = image
    return metadata


def parameters_to_generated_image_metadata(parameters):
    # top-level metadata minus `image` or `images`

    metadata = get_system_config()
    # remove any image keys not mentioned in RFC #266
    rfc266_img_fields = [
        "type",
        "postprocessing",
        "sampler",
        "prompt",
        "seed",
        "variations",
        "steps",
        "cfg_scale",
        "threshold",
        "perlin",
        "step_number",
        "width",
        "height",
        "extra",
        "seamless",
    ]

    rfc_dict = {}

    for item in parameters.items():
        key, value = item
        if key in rfc266_img_fields:
            rfc_dict[key] = value

    postprocessing = []

    # 'postprocessing' is either null or an
    if "gfpgan_strength" in parameters:

        postprocessing.append(
            {"type": "gfpgan", "strength": float(parameters["gfpgan_strength"])}
        )

    if "upscale" in parameters:
        postprocessing.append(
            {
                "type": "esrgan",
                "scale": int(parameters["upscale"][0]),
                "strength": float(parameters["upscale"][1]),
            }
        )

    rfc_dict["postprocessing"] = postprocessing if len(postprocessing) > 0 else None

    # semantic drift
    rfc_dict["sampler"] = parameters["sampler_name"]

    # display weighted subprompts (liable to change)
    subprompts = split_weighted_subprompts(parameters["prompt"])
    subprompts = [{"prompt": x[0], "weight": x[1]} for x in subprompts]
    rfc_dict["prompt"] = subprompts

    # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
    variations = []

    if "with_variations" in parameters:
        variations = [
            {"seed": x[0], "weight": x[1]} for x in parameters["with_variations"]
        ]

    rfc_dict["variations"] = variations

    if "init_img" in parameters:
        rfc_dict["type"] = "img2img"
        rfc_dict["strength"] = parameters["strength"]
        rfc_dict["fit"] = parameters["fit"]  # TODO: Noncompliant
        rfc_dict["orig_hash"] = calculate_init_img_hash(parameters["init_img"])
        rfc_dict["init_image_path"] = parameters["init_img"]  # TODO: Noncompliant
        rfc_dict["sampler"] = "ddim"  # TODO: FIX ME WHEN IMG2IMG SUPPORTS ALL SAMPLERS
        if "init_mask" in parameters:
            rfc_dict["mask_hash"] = calculate_init_img_hash(
                parameters["init_mask"]
            )  # TODO: Noncompliant
            rfc_dict["mask_image_path"] = parameters["init_mask"]  # TODO: Noncompliant
    else:
        rfc_dict["type"] = "txt2img"

    metadata["image"] = rfc_dict

    return metadata


def make_unique_init_image_filename(name):
    uuid = uuid4().hex
    split = os.path.splitext(name)
    name = f"{split[0]}.{uuid}{split[1]}"
    return name


def write_log_message(message, log_path=log_path):
    """Logs the filename and parameters used to generate or process that image to log file"""
    message = f"{message}\n"
    with open(log_path, "a", encoding="utf-8") as file:
        file.writelines(message)


def save_image(
    image, command, metadata, output_dir, step_index=None, postprocessing=False
):
    pngwriter = PngWriter(output_dir)
    prefix = pngwriter.unique_prefix()

    seed = "unknown_seed"

    if "image" in metadata:
        if "seed" in metadata["image"]:
            seed = metadata["image"]["seed"]

    filename = f"{prefix}.{seed}"

    if step_index:
        filename += f".{step_index}"
    if postprocessing:
        filename += f".postprocessed"

    filename += ".png"

    path = pngwriter.save_image_and_prompt_to_png(
        image=image, dream_prompt=command, metadata=metadata, name=filename
    )

    return path


def calculate_real_steps(steps, strength, has_init_image):
    return math.floor(strength * steps) if has_init_image else steps


def generate_images(generation_parameters, esrgan_parameters, gfpgan_parameters):
    canceled.clear()

    step_index = 1
    prior_variations = (
        generation_parameters["with_variations"]
        if "with_variations" in generation_parameters
        else []
    )
    """
    If a result image is used as an init image, and then deleted, we will want to be
    able to use it as an init image in the future. Need to copy it.

    If the init/mask image doesn't exist in the init_image_path/mask_image_path,
    make a unique filename for it and copy it there.
    """
    if "init_img" in generation_parameters:
        filename = os.path.basename(generation_parameters["init_img"])
        if not os.path.exists(os.path.join(init_image_path, filename)):
            unique_filename = make_unique_init_image_filename(filename)
            new_path = os.path.join(init_image_path, unique_filename)
            shutil.copy(generation_parameters["init_img"], new_path)
            generation_parameters["init_img"] = new_path
        if "init_mask" in generation_parameters:
            filename = os.path.basename(generation_parameters["init_mask"])
            if not os.path.exists(os.path.join(mask_image_path, filename)):
                unique_filename = make_unique_init_image_filename(filename)
                new_path = os.path.join(init_image_path, unique_filename)
                shutil.copy(generation_parameters["init_img"], new_path)
                generation_parameters["init_mask"] = new_path

    totalSteps = calculate_real_steps(
        steps=generation_parameters["steps"],
        strength=generation_parameters["strength"]
        if "strength" in generation_parameters
        else None,
        has_init_image="init_img" in generation_parameters,
    )

    progress = {
        "currentStep": 1,
        "totalSteps": totalSteps,
        "currentIteration": 1,
        "totalIterations": generation_parameters["iterations"],
        "currentStatus": "Preparing",
        "isProcessing": True,
        "currentStatusHasSteps": False,
    }

    socketio.emit("progressUpdate", progress)
    eventlet.sleep(0)

    def image_progress(sample, step):
        if canceled.is_set():
            raise CanceledException

        nonlocal step_index
        nonlocal generation_parameters
        nonlocal progress

        progress["currentStep"] = step + 1
        progress["currentStatus"] = "Generating"
        progress["currentStatusHasSteps"] = True

        if (
            generation_parameters["progress_images"]
            and step % 5 == 0
            and step < generation_parameters["steps"] - 1
        ):
            image = generate.sample_to_image(sample)

            metadata = parameters_to_generated_image_metadata(generation_parameters)
            command = parameters_to_command(generation_parameters)
            path = save_image(image, command, metadata, intermediate_path, step_index=step_index, postprocessing=False)

            step_index += 1
            socketio.emit(
                "intermediateResult",
                {
                    "url": os.path.relpath(path),
                    "mtime": os.path.getmtime(path),
                    "metadata": metadata,
                },
            )
        socketio.emit("progressUpdate", progress)
        eventlet.sleep(0)

    def image_done(image, seed, first_seed):
        nonlocal generation_parameters
        nonlocal esrgan_parameters
        nonlocal gfpgan_parameters
        nonlocal progress

        step_index = 1
        nonlocal prior_variations

        progress["currentStatus"] = "Generation complete"
        socketio.emit("progressUpdate", progress)
        eventlet.sleep(0)

        all_parameters = generation_parameters
        postprocessing = False

        if (
            "variation_amount" in all_parameters
            and all_parameters["variation_amount"] > 0
        ):
            first_seed = first_seed or seed
            this_variation = [[seed, all_parameters["variation_amount"]]]
            all_parameters["with_variations"] = prior_variations + this_variation
            all_parameters["seed"] = first_seed
        elif ("with_variations" in all_parameters):
            all_parameters["seed"] = first_seed
        else:
            all_parameters["seed"] = seed

        if esrgan_parameters:
            progress["currentStatus"] = "Upscaling"
            progress["currentStatusHasSteps"] = False
            socketio.emit("progressUpdate", progress)
            eventlet.sleep(0)

            image = esrgan.process(
                image=image,
                upsampler_scale=esrgan_parameters["level"],
                strength=esrgan_parameters["strength"],
                seed=seed,
            )

            postprocessing = True
            all_parameters["upscale"] = [
                esrgan_parameters["level"],
                esrgan_parameters["strength"],
            ]

        if gfpgan_parameters:
            progress["currentStatus"] = "Fixing faces"
            progress["currentStatusHasSteps"] = False
            socketio.emit("progressUpdate", progress)
            eventlet.sleep(0)

            image = gfpgan.process(
                image=image, strength=gfpgan_parameters["strength"], seed=seed
            )
            postprocessing = True
            all_parameters["gfpgan_strength"] = gfpgan_parameters["strength"]

        progress["currentStatus"] = "Saving image"
        socketio.emit("progressUpdate", progress)
        eventlet.sleep(0)

        metadata = parameters_to_generated_image_metadata(all_parameters)
        command = parameters_to_command(all_parameters)

        path = save_image(
            image, command, metadata, result_path, postprocessing=postprocessing
        )

        print(f'>> Image generated: "{path}"')
        write_log_message(f'[Generated] "{path}": {command}')

        if progress["totalIterations"] > progress["currentIteration"]:
            progress["currentStep"] = 1
            progress["currentIteration"] += 1
            progress["currentStatus"] = "Iteration finished"
            progress["currentStatusHasSteps"] = False
        else:
            progress["currentStep"] = 0
            progress["totalSteps"] = 0
            progress["currentIteration"] = 0
            progress["totalIterations"] = 0
            progress["currentStatus"] = "Finished"
            progress["isProcessing"] = False

        socketio.emit("progressUpdate", progress)
        eventlet.sleep(0)

        socketio.emit(
            "generationResult",
            {
                "url": os.path.relpath(path),
                "mtime": os.path.getmtime(path),
                "metadata": metadata,
            },
        )
        eventlet.sleep(0)

    try:
        generate.prompt2image(
            **generation_parameters,
            step_callback=image_progress,
            image_callback=image_done,
        )

    except KeyboardInterrupt:
        raise
    except CanceledException:
        pass
    except Exception as e:
        socketio.emit("error", {"message": (str(e))})
        print("\n")
        traceback.print_exc()
        print("\n")


"""
END ADDITIONAL FUNCTIONS
"""


if __name__ == "__main__":
    print(f">> Starting server at http://{host}:{port}")
    socketio.run(app, host=host, port=port)
