import eventlet
import glob
import os
import shutil
import mimetypes
import traceback
import math

from flask import Flask, redirect, send_from_directory
from flask_socketio import SocketIO
from PIL import Image
from uuid import uuid4
from threading import Event

from ldm.invoke.args import Args, APP_ID, APP_VERSION, calculate_init_img_hash
from ldm.invoke.pngwriter import PngWriter, retrieve_metadata
from ldm.invoke.conditioning import split_weighted_subprompts

from backend.modules.parameters import parameters_to_command


# Loading Arguments
opt = Args()
args = opt.parse_args()


class InvokeAIWebServer:
    def __init__(self, generate, gfpgan, codeformer, esrgan) -> None:
        self.host = args.host
        self.port = args.port

        self.generate = generate
        self.gfpgan = gfpgan
        self.codeformer = codeformer
        self.esrgan = esrgan

        self.canceled = Event()

    def run(self):
        self.setup_app()
        self.setup_flask()

    def setup_flask(self):
        # Fix missing mimetypes on Windows
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('text/css', '.css')
        # Socket IO
        logger = True if args.web_verbose else False
        engineio_logger = True if args.web_verbose else False
        max_http_buffer_size = 10000000

        socketio_args = {
            'logger': logger,
            'engineio_logger': engineio_logger,
            'max_http_buffer_size': max_http_buffer_size,
            'ping_interval': (50, 50),
            'ping_timeout': 60,
        }

        if opt.cors:
            socketio_args['cors_allowed_origins'] = opt.cors

        self.app = Flask(
            __name__, static_url_path='', static_folder='../frontend/dist/'
        )

        self.socketio = SocketIO(
            self.app,
            **socketio_args
        )

        # Keep Server Alive Route
        @self.app.route('/flaskwebgui-keep-server-alive')
        def keep_alive():
            return {'message': 'Server Running'}

        # Outputs Route
        self.app.config['OUTPUTS_FOLDER'] = os.path.abspath(args.outdir)

        @self.app.route('/outputs/<path:file_path>')
        def outputs(file_path):
            return send_from_directory(
                self.app.config['OUTPUTS_FOLDER'], file_path
            )

        # Base Route
        @self.app.route('/')
        def serve():
            if args.web_develop:
                return redirect('http://127.0.0.1:5173')
            else:
                return send_from_directory(
                    self.app.static_folder, 'index.html'
                )

        self.load_socketio_listeners(self.socketio)

        if args.gui:
            print('>> Launching Invoke AI GUI')
            close_server_on_exit = True
            if args.web_develop:
                close_server_on_exit = False
            try:
                from flaskwebgui import FlaskUI
                FlaskUI(
                    app=self.app,
                    socketio=self.socketio,
                    start_server='flask-socketio',
                    host=self.host,
                    port=self.port,
                    width=1600,
                    height=1000,
                    idle_interval=10,
                    close_server_on_exit=close_server_on_exit,
                ).run()
            except KeyboardInterrupt:
                import sys

                sys.exit(0)
        else:
            print('>> Started Invoke AI Web Server!')
            if self.host == '0.0.0.0':
                print(
                    f"Point your browser at http://localhost:{self.port} or use the host's DNS name or IP address."
                )
            else:
                print(
                    '>> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address.'
                )
                print(
                    f'>> Point your browser at http://{self.host}:{self.port}'
                )
            self.socketio.run(app=self.app, host=self.host, port=self.port)

    def setup_app(self):
        self.result_url = 'outputs/'
        self.init_image_url = 'outputs/init-images/'
        self.mask_image_url = 'outputs/mask-images/'
        self.intermediate_url = 'outputs/intermediates/'
        # location for "finished" images
        self.result_path = args.outdir
        # temporary path for intermediates
        self.intermediate_path = os.path.join(
            self.result_path, 'intermediates/'
        )
        # path for user-uploaded init images and masks
        self.init_image_path = os.path.join(self.result_path, 'init-images/')
        self.mask_image_path = os.path.join(self.result_path, 'mask-images/')
        # txt log
        self.log_path = os.path.join(self.result_path, 'invoke_log.txt')
        # make all output paths
        [
            os.makedirs(path, exist_ok=True)
            for path in [
                self.result_path,
                self.intermediate_path,
                self.init_image_path,
                self.mask_image_path,
            ]
        ]

    def load_socketio_listeners(self, socketio):
        @socketio.on('requestSystemConfig')
        def handle_request_capabilities():
            print(f'>> System config requested')
            config = self.get_system_config()
            socketio.emit('systemConfig', config)

        @socketio.on('requestLatestImages')
        def handle_request_latest_images(latest_mtime):
            try:
                paths = glob.glob(os.path.join(self.result_path, '*.png'))

                image_paths = sorted(
                    paths, key=lambda x: os.path.getmtime(x), reverse=True
                )

                image_paths = list(
                    filter(
                        lambda x: os.path.getmtime(x) > latest_mtime,
                        image_paths,
                    )
                )

                image_array = []

                for path in image_paths:
                    metadata = retrieve_metadata(path)
                    image_array.append(
                        {
                            'url': self.get_url_from_image_path(path),
                            'mtime': os.path.getmtime(path),
                            'metadata': metadata['sd-metadata'],
                        }
                    )

                socketio.emit(
                    'galleryImages',
                    {
                        'images': image_array,
                    },
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        @socketio.on('requestImages')
        def handle_request_images(earliest_mtime=None):
            try:
                page_size = 50

                paths = glob.glob(os.path.join(self.result_path, '*.png'))

                image_paths = sorted(
                    paths, key=lambda x: os.path.getmtime(x), reverse=True
                )

                if earliest_mtime:
                    image_paths = list(
                        filter(
                            lambda x: os.path.getmtime(x) < earliest_mtime,
                            image_paths,
                        )
                    )

                areMoreImagesAvailable = len(image_paths) >= page_size
                image_paths = image_paths[slice(0, page_size)]

                image_array = []

                for path in image_paths:
                    metadata = retrieve_metadata(path)
                    image_array.append(
                        {
                            'url': self.get_url_from_image_path(path),
                            'mtime': os.path.getmtime(path),
                            'metadata': metadata['sd-metadata'],
                        }
                    )

                socketio.emit(
                    'galleryImages',
                    {
                        'images': image_array,
                        'areMoreImagesAvailable': areMoreImagesAvailable,
                    },
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        @socketio.on('generateImage')
        def handle_generate_image_event(
            generation_parameters, esrgan_parameters, gfpgan_parameters
        ):
            try:
                print(
                    f'>> Image generation requested: {generation_parameters}\nESRGAN parameters: {esrgan_parameters}\nGFPGAN parameters: {gfpgan_parameters}'
                )
                self.generate_images(
                    generation_parameters, esrgan_parameters, gfpgan_parameters
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        @socketio.on('runPostprocessing')
        def handle_run_postprocessing(
            original_image, postprocessing_parameters
        ):
            try:
                print(
                    f'>> Postprocessing requested for "{original_image["url"]}": {postprocessing_parameters}'
                )

                progress = Progress()

                socketio.emit('progressUpdate', progress.to_formatted_dict())
                eventlet.sleep(0)

                original_image_path = self.get_image_path_from_url(
                    original_image['url']
                )

                image = Image.open(original_image_path)

                seed = (
                    original_image['metadata']['seed']
                    if 'seed' in original_image['metadata']
                    else 'unknown_seed'
                )

                if postprocessing_parameters['type'] == 'esrgan':
                    progress.set_current_status('Upscaling')
                elif postprocessing_parameters['type'] == 'gfpgan':
                    progress.set_current_status('Restoring Faces')

                socketio.emit('progressUpdate', progress.to_formatted_dict())
                eventlet.sleep(0)

                if postprocessing_parameters['type'] == 'esrgan':
                    image = self.esrgan.process(
                        image=image,
                        upsampler_scale=postprocessing_parameters['upscale'][
                            0
                        ],
                        strength=postprocessing_parameters['upscale'][1],
                        seed=seed,
                    )
                elif postprocessing_parameters['type'] == 'gfpgan':
                    image = self.gfpgan.process(
                        image=image,
                        strength=postprocessing_parameters['gfpgan_strength'],
                        seed=seed,
                    )
                else:
                    raise TypeError(
                        f'{postprocessing_parameters["type"]} is not a valid postprocessing type'
                    )

                progress.set_current_status('Saving Image')
                socketio.emit('progressUpdate', progress.to_formatted_dict())
                eventlet.sleep(0)

                postprocessing_parameters['seed'] = seed
                metadata = self.parameters_to_post_processed_image_metadata(
                    parameters=postprocessing_parameters,
                    original_image_path=original_image_path,
                )

                command = parameters_to_command(postprocessing_parameters)

                path = self.save_result_image(
                    image,
                    command,
                    metadata,
                    self.result_path,
                    postprocessing=postprocessing_parameters['type'],
                )

                self.write_log_message(
                    f'[Postprocessed] "{original_image_path}" > "{path}": {postprocessing_parameters}'
                )

                progress.mark_complete()
                socketio.emit('progressUpdate', progress.to_formatted_dict())
                eventlet.sleep(0)

                socketio.emit(
                    'postprocessingResult',
                    {
                        'url': self.get_url_from_image_path(path),
                        'mtime': os.path.getmtime(path),
                        'metadata': metadata,
                    },
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        @socketio.on('cancel')
        def handle_cancel():
            print(f'>> Cancel processing requested')
            self.canceled.set()

        # TODO: I think this needs a safety mechanism.
        @socketio.on('deleteImage')
        def handle_delete_image(url, uuid):
            try:
                print(f'>> Delete requested "{url}"')
                from send2trash import send2trash

                path = self.get_image_path_from_url(url)
                send2trash(path)
                socketio.emit('imageDeleted', {'url': url, 'uuid': uuid})
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        # TODO: I think this needs a safety mechanism.
        @socketio.on('uploadInitialImage')
        def handle_upload_initial_image(bytes, name):
            try:
                print(f'>> Init image upload requested "{name}"')
                file_path = self.save_file_unique_uuid_name(
                    bytes=bytes, name=name, path=self.init_image_path
                )

                socketio.emit(
                    'initialImageUploaded',
                    {
                        'url': self.get_url_from_image_path(file_path),
                    },
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

        # TODO: I think this needs a safety mechanism.
        @socketio.on('uploadMaskImage')
        def handle_upload_mask_image(bytes, name):
            try:
                print(f'>> Mask image upload requested "{name}"')

                file_path = self.save_file_unique_uuid_name(
                    bytes=bytes, name=name, path=self.mask_image_path
                )

                socketio.emit(
                    'maskImageUploaded',
                    {
                        'url': self.get_url_from_image_path(file_path),
                    },
                )
            except Exception as e:
                self.socketio.emit('error', {'message': (str(e))})
                print('\n')

                traceback.print_exc()
                print('\n')

    # App Functions
    def get_system_config(self):
        return {
            'model': 'stable diffusion',
            'model_id': args.model,
            'model_hash': self.generate.model_hash,
            'app_id': APP_ID,
            'app_version': APP_VERSION,
        }

    def generate_images(
        self, generation_parameters, esrgan_parameters, gfpgan_parameters
    ):
        try:
            self.canceled.clear()

            step_index = 1
            prior_variations = (
                generation_parameters['with_variations']
                if 'with_variations' in generation_parameters
                else []
            )

            """
            TODO:
            If a result image is used as an init image, and then deleted, we will want to be
            able to use it as an init image in the future. Need to handle this case.
            """

            # We need to give absolute paths to the generator, stash the URLs for later
            init_img_url = None
            mask_img_url = None

            if 'init_img' in generation_parameters:
                init_img_url = generation_parameters['init_img']
                generation_parameters[
                    'init_img'
                ] = self.get_image_path_from_url(
                    generation_parameters['init_img']
                )

            if 'init_mask' in generation_parameters:
                mask_img_url = generation_parameters['init_mask']
                generation_parameters[
                    'init_mask'
                ] = self.get_image_path_from_url(
                    generation_parameters['init_mask']
                )

            totalSteps = self.calculate_real_steps(
                steps=generation_parameters['steps'],
                strength=generation_parameters['strength']
                if 'strength' in generation_parameters
                else None,
                has_init_image='init_img' in generation_parameters,
            )

            progress = Progress(generation_parameters=generation_parameters)

            self.socketio.emit('progressUpdate', progress.to_formatted_dict())
            eventlet.sleep(0)

            def image_progress(sample, step):
                if self.canceled.is_set():
                    raise CanceledException

                nonlocal step_index
                nonlocal generation_parameters
                nonlocal progress

                progress.set_current_step(step + 1)
                progress.set_current_status('Generating')
                progress.set_current_status_has_steps(True)

                if (
                    generation_parameters['progress_images']
                    and step % 5 == 0
                    and step < generation_parameters['steps'] - 1
                ):
                    image = self.generate.sample_to_image(sample)
                    metadata = self.parameters_to_generated_image_metadata(
                        generation_parameters
                    )
                    command = parameters_to_command(generation_parameters)

                    path = self.save_result_image(
                        image,
                        command,
                        metadata,
                        self.intermediate_path,
                        step_index=step_index,
                        postprocessing=False,
                    )

                    step_index += 1
                    self.socketio.emit(
                        'intermediateResult',
                        {
                            'url': self.get_url_from_image_path(path),
                            'mtime': os.path.getmtime(path),
                            'metadata': metadata,
                        },
                    )
                self.socketio.emit(
                    'progressUpdate', progress.to_formatted_dict()
                )
                eventlet.sleep(0)

            def image_done(image, seed, first_seed):
                if self.canceled.is_set():
                    raise CanceledException

                nonlocal generation_parameters
                nonlocal esrgan_parameters
                nonlocal gfpgan_parameters
                nonlocal progress

                step_index = 1
                nonlocal prior_variations

                progress.set_current_status('Generation Complete')

                self.socketio.emit(
                    'progressUpdate', progress.to_formatted_dict()
                )
                eventlet.sleep(0)

                all_parameters = generation_parameters
                postprocessing = False

                if (
                    'variation_amount' in all_parameters
                    and all_parameters['variation_amount'] > 0
                ):
                    first_seed = first_seed or seed
                    this_variation = [
                        [seed, all_parameters['variation_amount']]
                    ]
                    all_parameters['with_variations'] = (
                        prior_variations + this_variation
                    )
                    all_parameters['seed'] = first_seed
                elif 'with_variations' in all_parameters:
                    all_parameters['seed'] = first_seed
                else:
                    all_parameters['seed'] = seed

                if self.canceled.is_set():
                    raise CanceledException

                if esrgan_parameters:
                    progress.set_current_status('Upscaling')
                    progress.set_current_status_has_steps(False)
                    self.socketio.emit(
                        'progressUpdate', progress.to_formatted_dict()
                    )
                    eventlet.sleep(0)

                    image = self.esrgan.process(
                        image=image,
                        upsampler_scale=esrgan_parameters['level'],
                        strength=esrgan_parameters['strength'],
                        seed=seed,
                    )

                    postprocessing = True
                    all_parameters['upscale'] = [
                        esrgan_parameters['level'],
                        esrgan_parameters['strength'],
                    ]

                if self.canceled.is_set():
                    raise CanceledException

                if gfpgan_parameters:
                    progress.set_current_status('Restoring Faces')
                    progress.set_current_status_has_steps(False)
                    self.socketio.emit(
                        'progressUpdate', progress.to_formatted_dict()
                    )
                    eventlet.sleep(0)

                    image = self.gfpgan.process(
                        image=image,
                        strength=gfpgan_parameters['strength'],
                        seed=seed,
                    )
                    postprocessing = True
                    all_parameters['gfpgan_strength'] = gfpgan_parameters[
                        'strength'
                    ]

                progress.set_current_status('Saving Image')
                self.socketio.emit(
                    'progressUpdate', progress.to_formatted_dict()
                )
                eventlet.sleep(0)

                # restore the stashed URLS and discard the paths, we are about to send the result to client
                if 'init_img' in all_parameters:
                    all_parameters['init_img'] = init_img_url

                if 'init_mask' in all_parameters:
                    all_parameters['init_mask'] = mask_img_url

                metadata = self.parameters_to_generated_image_metadata(
                    all_parameters
                )

                command = parameters_to_command(all_parameters)

                path = self.save_result_image(
                    image,
                    command,
                    metadata,
                    self.result_path,
                    postprocessing=postprocessing,
                )

                print(f'>> Image generated: "{path}"')
                self.write_log_message(f'[Generated] "{path}": {command}')

                if progress.total_iterations > progress.current_iteration:
                    progress.set_current_step(1)
                    progress.set_current_status('Iteration complete')
                    progress.set_current_status_has_steps(False)
                else:
                    progress.mark_complete()

                self.socketio.emit(
                    'progressUpdate', progress.to_formatted_dict()
                )
                eventlet.sleep(0)

                self.socketio.emit(
                    'generationResult',
                    {
                        'url': self.get_url_from_image_path(path),
                        'mtime': os.path.getmtime(path),
                        'metadata': metadata,
                    },
                )
                eventlet.sleep(0)

                progress.set_current_iteration(progress.current_iteration + 1)

            self.generate.prompt2image(
                **generation_parameters,
                step_callback=image_progress,
                image_callback=image_done,
            )

        except KeyboardInterrupt:
            raise
        except CanceledException:
            self.socketio.emit('processingCanceled')
            pass
        except Exception as e:
            print(e)
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def parameters_to_generated_image_metadata(self, parameters):
        try:
            # top-level metadata minus `image` or `images`
            metadata = self.get_system_config()
            # remove any image keys not mentioned in RFC #266
            rfc266_img_fields = [
                'type',
                'postprocessing',
                'sampler',
                'prompt',
                'seed',
                'variations',
                'steps',
                'cfg_scale',
                'threshold',
                'perlin',
                'step_number',
                'width',
                'height',
                'extra',
                'seamless',
            ]

            rfc_dict = {}

            for item in parameters.items():
                key, value = item
                if key in rfc266_img_fields:
                    rfc_dict[key] = value

            postprocessing = []

            # 'postprocessing' is either null or an
            if 'gfpgan_strength' in parameters:

                postprocessing.append(
                    {
                        'type': 'gfpgan',
                        'strength': float(parameters['gfpgan_strength']),
                    }
                )

            if 'upscale' in parameters:
                postprocessing.append(
                    {
                        'type': 'esrgan',
                        'scale': int(parameters['upscale'][0]),
                        'strength': float(parameters['upscale'][1]),
                    }
                )

            rfc_dict['postprocessing'] = (
                postprocessing if len(postprocessing) > 0 else None
            )

            # semantic drift
            rfc_dict['sampler'] = parameters['sampler_name']

            # display weighted subprompts (liable to change)
            subprompts = split_weighted_subprompts(parameters['prompt'])
            subprompts = [{'prompt': x[0], 'weight': x[1]} for x in subprompts]
            rfc_dict['prompt'] = subprompts

            # 'variations' should always exist and be an array, empty or consisting of {'seed': seed, 'weight': weight} pairs
            variations = []

            if 'with_variations' in parameters:
                variations = [
                    {'seed': x[0], 'weight': x[1]}
                    for x in parameters['with_variations']
                ]

            rfc_dict['variations'] = variations

            if 'init_img' in parameters:
                rfc_dict['type'] = 'img2img'
                rfc_dict['strength'] = parameters['strength']
                rfc_dict['fit'] = parameters['fit']  # TODO: Noncompliant
                rfc_dict['orig_hash'] = calculate_init_img_hash(
                    self.get_image_path_from_url(parameters['init_img'])
                )
                rfc_dict['init_image_path'] = parameters[
                    'init_img'
                ]  # TODO: Noncompliant
                if 'init_mask' in parameters:
                    rfc_dict['mask_hash'] = calculate_init_img_hash(
                        self.get_image_path_from_url(parameters['init_mask'])
                    )  # TODO: Noncompliant
                    rfc_dict['mask_image_path'] = parameters[
                        'init_mask'
                    ]  # TODO: Noncompliant
            else:
                rfc_dict['type'] = 'txt2img'

            metadata['image'] = rfc_dict

            return metadata

        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def parameters_to_post_processed_image_metadata(
        self, parameters, original_image_path
    ):
        try:
            current_metadata = retrieve_metadata(original_image_path)[
                'sd-metadata'
            ]
            postprocessing_metadata = {}

            """
            if we don't have an original image metadata to reconstruct,
            need to record the original image and its hash
            """
            if 'image' not in current_metadata:
                current_metadata['image'] = {}

                orig_hash = calculate_init_img_hash(
                    self.get_image_path_from_url(original_image_path)
                )

                postprocessing_metadata['orig_path'] = (original_image_path,)
                postprocessing_metadata['orig_hash'] = orig_hash

            if parameters['type'] == 'esrgan':
                postprocessing_metadata['type'] = 'esrgan'
                postprocessing_metadata['scale'] = parameters['upscale'][0]
                postprocessing_metadata['strength'] = parameters['upscale'][1]
            elif parameters['type'] == 'gfpgan':
                postprocessing_metadata['type'] = 'gfpgan'
                postprocessing_metadata['strength'] = parameters[
                    'gfpgan_strength'
                ]
            else:
                raise TypeError(f"Invalid type: {parameters['type']}")

            if 'postprocessing' in current_metadata['image'] and isinstance(
                current_metadata['image']['postprocessing'], list
            ):
                current_metadata['image']['postprocessing'].append(
                    postprocessing_metadata
                )
            else:
                current_metadata['image']['postprocessing'] = [
                    postprocessing_metadata
                ]

            return current_metadata

        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def save_result_image(
        self,
        image,
        command,
        metadata,
        output_dir,
        step_index=None,
        postprocessing=False,
    ):
        try:
            pngwriter = PngWriter(output_dir)
            prefix = pngwriter.unique_prefix()

            seed = 'unknown_seed'

            if 'image' in metadata:
                if 'seed' in metadata['image']:
                    seed = metadata['image']['seed']

            filename = f'{prefix}.{seed}'

            if step_index:
                filename += f'.{step_index}'
            if postprocessing:
                filename += f'.postprocessed'

            filename += '.png'

            path = pngwriter.save_image_and_prompt_to_png(
                image=image,
                dream_prompt=command,
                metadata=metadata,
                name=filename,
            )

            return os.path.abspath(path)

        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def make_unique_init_image_filename(self, name):
        try:
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            return name
        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def calculate_real_steps(self, steps, strength, has_init_image):
        import math

        return math.floor(strength * steps) if has_init_image else steps

    def write_log_message(self, message):
        """Logs the filename and parameters used to generate or process that image to log file"""
        try:
            message = f'{message}\n'
            with open(self.log_path, 'a', encoding='utf-8') as file:
                file.writelines(message)

        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def get_image_path_from_url(self, url):
        """Given a url to an image used by the client, returns the absolute file path to that image"""
        try:
            if 'init-images' in url:
                return os.path.abspath(
                    os.path.join(self.init_image_path, os.path.basename(url))
                )
            elif 'mask-images' in url:
                return os.path.abspath(
                    os.path.join(self.mask_image_path, os.path.basename(url))
                )
            elif 'intermediates' in url:
                return os.path.abspath(
                    os.path.join(self.intermediate_path, os.path.basename(url))
                )
            else:
                return os.path.abspath(
                    os.path.join(self.result_path, os.path.basename(url))
                )
        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def get_url_from_image_path(self, path):
        """Given an absolute file path to an image, returns the URL that the client can use to load the image"""
        try:
            if 'init-images' in path:
                return os.path.join(
                    self.init_image_url, os.path.basename(path)
                )
            elif 'mask-images' in path:
                return os.path.join(
                    self.mask_image_url, os.path.basename(path)
                )
            elif 'intermediates' in path:
                return os.path.join(
                    self.intermediate_url, os.path.basename(path)
                )
            else:
                return os.path.join(self.result_url, os.path.basename(path))
        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')

    def save_file_unique_uuid_name(self, bytes, name, path):
        try:
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            file_path = os.path.join(path, name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            newFile = open(file_path, 'wb')
            newFile.write(bytes)
            return file_path
        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')

            traceback.print_exc()
            print('\n')


class Progress:
    def __init__(self, generation_parameters=None):
        self.current_step = 1
        self.total_steps = (
            self._calculate_real_steps(
                steps=generation_parameters['steps'],
                strength=generation_parameters['strength']
                if 'strength' in generation_parameters
                else None,
                has_init_image='init_img' in generation_parameters,
            )
            if generation_parameters
            else 1
        )
        self.current_iteration = 1
        self.total_iterations = (
            generation_parameters['iterations'] if generation_parameters else 1
        )
        self.current_status = 'Preparing'
        self.is_processing = True
        self.current_status_has_steps = False
        self.has_error = False

    def set_current_step(self, current_step):
        self.current_step = current_step

    def set_total_steps(self, total_steps):
        self.total_steps = total_steps

    def set_current_iteration(self, current_iteration):
        self.current_iteration = current_iteration

    def set_total_iterations(self, total_iterations):
        self.total_iterations = total_iterations

    def set_current_status(self, current_status):
        self.current_status = current_status

    def set_is_processing(self, is_processing):
        self.is_processing = is_processing

    def set_current_status_has_steps(self, current_status_has_steps):
        self.current_status_has_steps = current_status_has_steps

    def set_has_error(self, has_error):
        self.has_error = has_error

    def mark_complete(self):
        self.current_status = 'Processing Complete'
        self.current_step = 0
        self.total_steps = 0
        self.current_iteration = 0
        self.total_iterations = 0
        self.is_processing = False

    def to_formatted_dict(
        self,
    ):
        return {
            'currentStep': self.current_step,
            'totalSteps': self.total_steps,
            'currentIteration': self.current_iteration,
            'totalIterations': self.total_iterations,
            'currentStatus': self.current_status,
            'isProcessing': self.is_processing,
            'currentStatusHasSteps': self.current_status_has_steps,
            'hasError': self.has_error,
        }

    def _calculate_real_steps(self, steps, strength, has_init_image):
        return math.floor(strength * steps) if has_init_image else steps


class CanceledException(Exception):
    pass
