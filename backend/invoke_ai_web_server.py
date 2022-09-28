import eventlet
import glob
import os
import shutil
import mimetypes

from flask import Flask, redirect, send_from_directory
from flask_socketio import SocketIO
from PIL import Image
from uuid import uuid4
from threading import Event

from ldm.dream.args import Args, APP_ID, APP_VERSION, calculate_init_img_hash
from ldm.dream.pngwriter import PngWriter, retrieve_metadata
from ldm.dream.conditioning import split_weighted_subprompts

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
        mimetypes.add_type("application/javascript", ".js")
        mimetypes.add_type("text/css", ".css")
        # Socket IO
        logger = True if args.web_verbose else False
        engineio_logger = True if args.web_verbose else False
        max_http_buffer_size = 10000000

        # CORS Allowed Setup
        cors_allowed_origins = ['http://127.0.0.1:5173', 'http://localhost:5173']
        additional_allowed_origins = (
            opt.cors if opt.cors else []
        )  # additional CORS allowed origins
        if self.host == '127.0.0.1':
            cors_allowed_origins.extend(
                [
                    f'http://{self.host}:{self.port}',
                    f'http://localhost:{self.port}',
                ]
            )
        cors_allowed_origins = (
            cors_allowed_origins + additional_allowed_origins
        )

        self.app = Flask(
            __name__, static_url_path='', static_folder='../frontend/dist/'
        )

        self.socketio = SocketIO(
            self.app,
            logger=logger,
            engineio_logger=engineio_logger,
            max_http_buffer_size=max_http_buffer_size,
            cors_allowed_origins=cors_allowed_origins,
            ping_interval=(50, 50),
            ping_timeout=60,
        )


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

        print('>> Started Invoke AI Web Server!')
        if self.host == '0.0.0.0':
            print(
                f"Point your browser at http://localhost:{self.port} or use the host's DNS name or IP address."
            )
        else:
            print(
                '>> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address.'
            )
            print(f'>> Point your browser at http://{self.host}:{self.port}')

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
        self.log_path = os.path.join(self.result_path, 'dream_log.txt')
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

        @socketio.on('requestImages')
        def handle_request_images(page=1, offset=0, last_mtime=None):
            chunk_size = 50

            if last_mtime:
                print(f'>> Latest images requested')
            else:
                print(
                    f'>> Page {page} of images requested (page size {chunk_size} offset {offset})'
                )

            paths = glob.glob(os.path.join(self.result_path, '*.png'))
            sorted_paths = sorted(
                paths, key=lambda x: os.path.getmtime(x), reverse=True
            )

            if last_mtime:
                image_paths = filter(
                    lambda x: os.path.getmtime(x) > last_mtime, sorted_paths
                )
            else:

                image_paths = sorted_paths[
                    slice(
                        chunk_size * (page - 1) + offset,
                        chunk_size * page + offset,
                    )
                ]
                page = page + 1

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
                    'nextPage': page,
                    'offset': offset,
                    'onlyNewImages': True if last_mtime else False,
                },
            )

        @socketio.on('generateImage')
        def handle_generate_image_event(
            generation_parameters, esrgan_parameters, gfpgan_parameters
        ):
            print(
                f'>> Image generation requested: {generation_parameters}\nESRGAN parameters: {esrgan_parameters}\nGFPGAN parameters: {gfpgan_parameters}'
            )
            self.generate_images(
                generation_parameters, esrgan_parameters, gfpgan_parameters
            )

        @socketio.on('runESRGAN')
        def handle_run_esrgan_event(original_image, esrgan_parameters):
            print(
                f'>> ESRGAN upscale requested for "{original_image["url"]}": {esrgan_parameters}'
            )
            progress = {
                'currentStep': 1,
                'totalSteps': 1,
                'currentIteration': 1,
                'totalIterations': 1,
                'currentStatus': 'Preparing',
                'isProcessing': True,
                'currentStatusHasSteps': False,
            }

            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            original_image_path = self.get_image_path_from_url(original_image['url'])
            # os.path.join(self.result_path, os.path.basename(original_image['url']))

            image = Image.open(original_image_path)

            seed = (
                original_image['metadata']['seed']
                if 'seed' in original_image['metadata']
                else 'unknown_seed'
            )

            progress['currentStatus'] = 'Upscaling'
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            image = self.esrgan.process(
                image=image,
                upsampler_scale=esrgan_parameters['upscale'][0],
                strength=esrgan_parameters['upscale'][1],
                seed=seed,
            )

            progress['currentStatus'] = 'Saving image'
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            esrgan_parameters['seed'] = seed
            metadata = self.parameters_to_post_processed_image_metadata(
                parameters=esrgan_parameters,
                original_image_path=original_image_path,
                type='esrgan',
            )
            command = parameters_to_command(esrgan_parameters)

            path = self.save_image(
                image,
                command,
                metadata,
                self.result_path,
                postprocessing='esrgan',
            )

            self.write_log_message(
                f'[Upscaled] "{original_image_path}" > "{path}": {command}'
            )

            progress['currentStatus'] = 'Finished'
            progress['currentStep'] = 0
            progress['totalSteps'] = 0
            progress['currentIteration'] = 0
            progress['totalIterations'] = 0
            progress['isProcessing'] = False
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            socketio.emit(
                'esrganResult',
                {
                    'url': self.get_url_from_image_path(path),
                    'mtime': os.path.getmtime(path),
                    'metadata': metadata,
                },
            )

        @socketio.on('runGFPGAN')
        def handle_run_gfpgan_event(original_image, gfpgan_parameters):
            print(
                f'>> GFPGAN face fix requested for "{original_image["url"]}": {gfpgan_parameters}'
            )
            progress = {
                'currentStep': 1,
                'totalSteps': 1,
                'currentIteration': 1,
                'totalIterations': 1,
                'currentStatus': 'Preparing',
                'isProcessing': True,
                'currentStatusHasSteps': False,
            }

            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            original_image_path = self.get_image_path_from_url(original_image['url'])

            image = Image.open(original_image_path)

            seed = (
                original_image['metadata']['seed']
                if 'seed' in original_image['metadata']
                else 'unknown_seed'
            )

            progress['currentStatus'] = 'Fixing faces'
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            image = self.gfpgan.process(
                image=image,
                strength=gfpgan_parameters['gfpgan_strength'],
                seed=seed,
            )

            progress['currentStatus'] = 'Saving image'
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            gfpgan_parameters['seed'] = seed
            metadata = self.parameters_to_post_processed_image_metadata(
                parameters=gfpgan_parameters,
                original_image_path=original_image_path,
                type='gfpgan',
            )
            command = parameters_to_command(gfpgan_parameters)

            path = self.save_image(
                image,
                command,
                metadata,
                self.result_path,
                postprocessing='gfpgan',
            )

            self.write_log_message(
                f'[Fixed faces] "{original_image_path}" > "{path}": {command}'
            )

            progress['currentStatus'] = 'Finished'
            progress['currentStep'] = 0
            progress['totalSteps'] = 0
            progress['currentIteration'] = 0
            progress['totalIterations'] = 0
            progress['isProcessing'] = False
            socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            socketio.emit(
                'gfpganResult',
                {
                    'url': self.get_url_from_image_path(path),
                    'mtime': os.path.getmtime(path),
                    'metadata': metadata,
                },
            )

        @socketio.on('cancel')
        def handle_cancel():
            print(f'>> Cancel processing requested')
            self.canceled.set()
            socketio.emit('processingCanceled')

        # TODO: I think this needs a safety mechanism.
        @socketio.on('deleteImage')
        def handle_delete_image(path, uuid):
            print(f'>> Delete requested "{path}"')
            from send2trash import send2trash

            path = self.get_image_path_from_url(path)
            send2trash(path)
            socketio.emit('imageDeleted', {'url': path, 'uuid': uuid})

        # TODO: I think this needs a safety mechanism.
        @socketio.on('uploadInitialImage')
        def handle_upload_initial_image(bytes, name):
            print(f'>> Init image upload requested "{name}"')
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            file_path = os.path.join(self.init_image_path, name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            newFile = open(file_path, 'wb')
            newFile.write(bytes)

            socketio.emit(
                'initialImageUploaded', {'url': self.get_url_from_image_path(file_path), 'uuid': ''}
            )

        # TODO: I think this needs a safety mechanism.
        @socketio.on('uploadMaskImage')
        def handle_upload_mask_image(bytes, name):
            print(f'>> Mask image upload requested "{name}"')
            uuid = uuid4().hex
            split = os.path.splitext(name)
            name = f'{split[0]}.{uuid}{split[1]}'
            file_path = os.path.join(self.mask_image_path, name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            newFile = open(file_path, 'wb')
            newFile.write(bytes)

            socketio.emit('maskImageUploaded', {'url': self.get_url_from_image_path(file_path), 'uuid': ''})

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
        self.canceled.clear()

        step_index = 1
        prior_variations = (
            generation_parameters['with_variations']
            if 'with_variations' in generation_parameters
            else []
        )

        """
        TODO: RE-IMPLEMENT THE COMMENTED-OUT CODE
        If a result image is used as an init image, and then deleted, we will want to be
        able to use it as an init image in the future. Need to copy it.

        If the init/mask image doesn't exist in the init_image_path/mask_image_path,
        make a unique filename for it and copy it there.
        """
        # if 'init_img' in generation_parameters:
        #     filename = os.path.basename(generation_parameters['init_img'])
        #     abs_init_image_path = os.path.join(self.init_image_path, filename)
        #     if not os.path.exists(
        #         abs_init_image_path
        #     ):
        #         unique_filename = self.make_unique_init_image_filename(
        #             filename
        #         )
        #         new_path = os.path.join(self.init_image_path, unique_filename)
        #         shutil.copy(abs_init_image_path, new_path)
        #         generation_parameters['init_img'] = os.path.abspath(new_path)
        #     else:
        #         generation_parameters['init_img'] = os.path.abspath(os.path.join(self.init_image_path, filename))

        #     if 'init_mask' in generation_parameters:
        #         filename = os.path.basename(generation_parameters['init_mask'])
        #         if not os.path.exists(
        #             os.path.join(self.mask_image_path, filename)
        #         ):
        #             unique_filename = self.make_unique_init_image_filename(
        #                 filename
        #             )
        #             new_path = os.path.join(
        #                 self.init_image_path, unique_filename
        #             )
        #             shutil.copy(generation_parameters['init_img'], new_path)
        #             generation_parameters['init_mask'] = os.path.abspath(new_path)
        #         else:
        #             generation_parameters['init_mas'] = os.path.abspath(os.path.join(self.mask_image_path, filename))


        # We need to give absolute paths to the generator, stash the URLs for later
        init_img_url =  None;
        mask_img_url =  None;

        if 'init_img' in generation_parameters:
            init_img_url = generation_parameters['init_img']
            generation_parameters['init_img'] = self.get_image_path_from_url(generation_parameters['init_img'])

        if 'init_mask' in generation_parameters:
            mask_img_url = generation_parameters['init_mask']
            generation_parameters['init_mask'] = self.get_image_path_from_url(generation_parameters['init_mask'])

        totalSteps = self.calculate_real_steps(
            steps=generation_parameters['steps'],
            strength=generation_parameters['strength']
            if 'strength' in generation_parameters
            else None,
            has_init_image='init_img' in generation_parameters,
        )

        progress = {
            'currentStep': 1,
            'totalSteps': totalSteps,
            'currentIteration': 1,
            'totalIterations': generation_parameters['iterations'],
            'currentStatus': 'Preparing',
            'isProcessing': True,
            'currentStatusHasSteps': False,
        }

        self.socketio.emit('progressUpdate', progress)
        eventlet.sleep(0)

        def image_progress(sample, step):
            if self.canceled.is_set():
                raise CanceledException

            nonlocal step_index
            nonlocal generation_parameters
            nonlocal progress

            progress['currentStep'] = step + 1
            progress['currentStatus'] = 'Generating'
            progress['currentStatusHasSteps'] = True

            if (
                generation_parameters['progress_images']
                and step % 5 == 0
                and step < generation_parameters['steps'] - 1
            ):
                image = self.generate.sample_to_image(sample)
                metadata = self.parameters_to_generated_image_metadata(generation_parameters)
                command = parameters_to_command(generation_parameters)

                path = self.save_image(image, command, metadata, self.intermediate_path, step_index=step_index, postprocessing=False)

                step_index += 1
                self.socketio.emit(
                    'intermediateResult',
                    {
                        'url': self.get_url_from_image_path(path),
                        'mtime': os.path.getmtime(path),
                        'metadata': metadata,
                    },
                )
            self.socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

        def image_done(image, seed, first_seed):
            nonlocal generation_parameters
            nonlocal esrgan_parameters
            nonlocal gfpgan_parameters
            nonlocal progress

            step_index = 1
            nonlocal prior_variations

            progress['currentStatus'] = 'Generation complete'
            self.socketio.emit('progressUpdate', progress)
            eventlet.sleep(0)

            all_parameters = generation_parameters
            postprocessing = False

            if (
                'variation_amount' in all_parameters
                and all_parameters['variation_amount'] > 0
            ):
                first_seed = first_seed or seed
                this_variation = [[seed, all_parameters['variation_amount']]]
                all_parameters['with_variations'] = (
                    prior_variations + this_variation
                )
                all_parameters['seed'] = first_seed
            elif 'with_variations' in all_parameters:
                all_parameters['seed'] = first_seed
            else:
                all_parameters['seed'] = seed

            if esrgan_parameters:
                progress['currentStatus'] = 'Upscaling'
                progress['currentStatusHasSteps'] = False
                self.socketio.emit('progressUpdate', progress)
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

            if gfpgan_parameters:
                progress['currentStatus'] = 'Fixing faces'
                progress['currentStatusHasSteps'] = False
                self.socketio.emit('progressUpdate', progress)
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

            progress['currentStatus'] = 'Saving image'
            self.socketio.emit('progressUpdate', progress)
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

            path = self.save_image(
                image,
                command,
                metadata,
                self.result_path,
                postprocessing=postprocessing,
            )

            print(f'>> Image generated: "{path}"')
            self.write_log_message(f'[Generated] "{path}": {command}')

            if progress['totalIterations'] > progress['currentIteration']:
                progress['currentStep'] = 1
                progress['currentIteration'] += 1
                progress['currentStatus'] = 'Iteration finished'
                progress['currentStatusHasSteps'] = False
            else:
                progress['currentStep'] = 0
                progress['totalSteps'] = 0
                progress['currentIteration'] = 0
                progress['totalIterations'] = 0
                progress['currentStatus'] = 'Finished'
                progress['isProcessing'] = False

            self.socketio.emit('progressUpdate', progress)
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

        try:
            self.generate.prompt2image(
                **generation_parameters,
                step_callback=image_progress,
                image_callback=image_done,
            )

        except KeyboardInterrupt:
            raise
        except CanceledException:
            pass
        except Exception as e:
            self.socketio.emit('error', {'message': (str(e))})
            print('\n')
            import traceback

            traceback.print_exc()
            print('\n')

    def parameters_to_generated_image_metadata(self, parameters):
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
            rfc_dict['orig_hash'] = calculate_init_img_hash(self.get_image_path_from_url(parameters['init_img']))
            rfc_dict['init_image_path'] = parameters[
                'init_img'
            ]  # TODO: Noncompliant
            rfc_dict[
                'sampler'
            ] = 'ddim'  # TODO: FIX ME WHEN IMG2IMG SUPPORTS ALL SAMPLERS
            if 'init_mask' in parameters:
                rfc_dict['mask_hash'] = calculate_init_img_hash(self.get_image_path_from_url(parameters['init_mask'])) # TODO: Noncompliant
                rfc_dict['mask_image_path'] = parameters[
                    'init_mask'
                ]  # TODO: Noncompliant
        else:
            rfc_dict['type'] = 'txt2img'

        metadata['image'] = rfc_dict

        return metadata

    def parameters_to_post_processed_image_metadata(
        self, parameters, original_image_path, type
    ):
        # top-level metadata minus `image` or `images`
        metadata = self.get_system_config()

        orig_hash = calculate_init_img_hash(self.get_image_path_from_url(original_image_path))

        image = {'orig_path': original_image_path, 'orig_hash': orig_hash}

        if type == 'esrgan':
            image['type'] = 'esrgan'
            image['scale'] = parameters['upscale'][0]
            image['strength'] = parameters['upscale'][1]
        elif type == 'gfpgan':
            image['type'] = 'gfpgan'
            image['strength'] = parameters['gfpgan_strength']
        else:
            raise TypeError(f'Invalid type: {type}')

        metadata['image'] = image
        return metadata

    def save_image(
        self,
        image,
        command,
        metadata,
        output_dir,
        step_index=None,
        postprocessing=False,
    ):
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
            image=image, dream_prompt=command, metadata=metadata, name=filename
        )

        return os.path.abspath(path)

    def make_unique_init_image_filename(self, name):
        uuid = uuid4().hex
        split = os.path.splitext(name)
        name = f'{split[0]}.{uuid}{split[1]}'
        return name

    def calculate_real_steps(self, steps, strength, has_init_image):
        import math
        return math.floor(strength * steps) if has_init_image else steps

    def write_log_message(self, message):
        """Logs the filename and parameters used to generate or process that image to log file"""
        message = f'{message}\n'
        with open(self.log_path, 'a', encoding='utf-8') as file:
            file.writelines(message)

    def get_image_path_from_url(self, url):
        """Given a url to an image used by the client, returns the absolute file path to that image"""
        if 'init-images' in url:
            return os.path.abspath(os.path.join(self.init_image_path, os.path.basename(url)))
        elif 'mask-images' in url:
            return os.path.abspath(os.path.join(self.mask_image_path, os.path.basename(url)))
        elif 'intermediates' in url:
            return os.path.abspath(os.path.join(self.intermediate_path, os.path.basename(url)))
        else:
            return os.path.abspath(os.path.join(self.result_path, os.path.basename(url)))

    def get_url_from_image_path(self, path):
        """Given an absolute file path to an image, returns the URL that the client can use to load the image"""
        if 'init-images' in path:
            return os.path.join(self.init_image_url, os.path.basename(path))
        elif 'mask-images' in path:
            return os.path.join(self.mask_image_url, os.path.basename(path))
        elif 'intermediates' in path:
            return os.path.join(self.intermediate_url, os.path.basename(path))
        else:
            return os.path.join(self.result_url, os.path.basename(path))



class CanceledException(Exception):
    pass