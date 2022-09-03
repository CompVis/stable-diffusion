#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import argparse
import shlex
import os
import re
import sys
import copy
import warnings
import time
import ldm.dream.readline
from ldm.dream.pngwriter import PngWriter, PromptFormatter
from ldm.dream.server import DreamServer, ThreadingDreamServer
from ldm.dream.image_util import make_grid
from omegaconf import OmegaConf

def main():
    """Initialize command-line parsers and the diffusion model"""
    arg_parser = create_argv_parser()
    opt = arg_parser.parse_args()
    
    if opt.laion400m:
        print('--laion400m flag has been deprecated. Please use --model laion400m instead.')
        sys.exit(-1)
    if opt.weights != 'model':
        print('--weights argument has been deprecated. Please configure ./configs/models.yaml, and call it using --model instead.')
        sys.exit(-1)
        
    try:
        models  = OmegaConf.load(opt.config)
        width   = models[opt.model].width
        height  = models[opt.model].height
        config  = models[opt.model].config
        weights = models[opt.model].weights
    except (FileNotFoundError, IOError, KeyError) as e:
        print(f'{e}. Aborting.')
        sys.exit(-1)

    print('* Initializing, be patient...\n')
    sys.path.append('.')
    from pytorch_lightning import logging
    from ldm.simplet2i import T2I

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers

    transformers.logging.set_verbosity_error()

    # creating a simple text2image object with a handful of
    # defaults passed on the command line.
    # additional parameters will be added (or overriden) during
    # the user input loop
    t2i = T2I(
        width=width,
        height=height,
        sampler_name=opt.sampler_name,
        weights=weights,
        full_precision=opt.full_precision,
        config=config,
        grid  = opt.grid,
        # this is solely for recreating the prompt
        latent_diffusion_weights=opt.laion400m,
        embedding_path=opt.embedding_path,
        device_type=opt.device
    )

    # make sure the output directory exists
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    # gets rid of annoying messages about random seed
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # load the infile as a list of lines
    infile = None
    if opt.infile:
        try:
            if os.path.isfile(opt.infile):
                infile = open(opt.infile, 'r', encoding='utf-8')
            elif opt.infile == '-':  # stdin
                infile = sys.stdin
            else:
                raise FileNotFoundError(f'{opt.infile} not found.')
        except (FileNotFoundError, IOError) as e:
            print(f'{e}. Aborting.')
            sys.exit(-1)

    # preload the model
    tic = time.time()
    t2i.load_model()
    print(
        f'>> model loaded in', '%4.2fs' % (time.time() - tic)
    )

    if not infile:
        print(
            "\n* Initialization done! Awaiting your command (-h for help, 'q' to quit)"
        )

    cmd_parser = create_cmd_parser()
    if opt.web:
        dream_server_loop(t2i, opt.host, opt.port)
    else:
        main_loop(t2i, opt.outdir, opt.prompt_as_dir, cmd_parser, infile)


def main_loop(t2i, outdir, prompt_as_dir, parser, infile):
    """prompt/read/execute loop"""
    done = False
    last_seeds = []
    path_filter = re.compile(r'[<>:"/\\|?*]')

    # os.pathconf is not available on Windows
    if hasattr(os, 'pathconf'):
        path_max = os.pathconf(outdir, 'PC_PATH_MAX')
        name_max = os.pathconf(outdir, 'PC_NAME_MAX')
    else:
        path_max = 260
        name_max = 255

    while not done:
        try:
            command = get_next_command(infile)
        except EOFError:
            done = True
            break

        # skip empty lines
        if not command.strip():
            continue

        if command.startswith(('#', '//')):
            continue

        # before splitting, escape single quotes so as not to mess
        # up the parser
        command = command.replace("'", "\\'")

        try:
            elements = shlex.split(command)
        except ValueError as e:
            print(str(e))
            continue

        if elements[0] == 'q':
            done = True
            break

        if elements[0].startswith(
            '!dream'
        ):   # in case a stored prompt still contains the !dream command
            elements.pop(0)

        # rearrange the arguments to mimic how it works in the Dream bot.
        switches = ['']
        switches_started = False

        for el in elements:
            if el[0] == '-' and not switches_started:
                switches_started = True
            if switches_started:
                switches.append(el)
            else:
                switches[0] += el
                switches[0] += ' '
        switches[0] = switches[0][: len(switches[0]) - 1]

        try:
            opt = parser.parse_args(switches)
        except SystemExit:
            parser.print_help()
            continue
        if len(opt.prompt) == 0:
            print('Try again with a prompt!')
            continue
        if opt.seed is not None and opt.seed < 0:   # retrieve previous value!
            try:
                opt.seed = last_seeds[opt.seed]
                print(f'reusing previous seed {opt.seed}')
            except IndexError:
                print(f'No previous seed at position {opt.seed} found')
                opt.seed = None

        do_grid           = opt.grid or t2i.grid

        if opt.with_variations is not None:
            # shotgun parsing, woo
            parts = []
            broken = False # python doesn't have labeled loops...
            for part in opt.with_variations.split(','):
                seed_and_weight = part.split(':')
                if len(seed_and_weight) != 2:
                    print(f'could not parse with_variation part "{part}"')
                    broken = True
                    break
                try:
                    seed = int(seed_and_weight[0])
                    weight = float(seed_and_weight[1])
                except ValueError:
                    print(f'could not parse with_variation part "{part}"')
                    broken = True
                    break
                parts.append([seed, weight])
            if broken:
                continue
            if len(parts) > 0:
                opt.with_variations = parts
            else:
                opt.with_variations = None

        if opt.outdir:
            if not os.path.exists(opt.outdir):
                os.makedirs(opt.outdir)
            current_outdir = opt.outdir
        elif prompt_as_dir:
            # sanitize the prompt to a valid folder name
            subdir = path_filter.sub('_', opt.prompt)[:name_max].rstrip(' .')

            # truncate path to maximum allowed length
            # 27 is the length of '######.##########.##.png', plus two separators and a NUL
            subdir = subdir[:(path_max - 27 - len(os.path.abspath(outdir)))]
            current_outdir = os.path.join(outdir, subdir)

            print ('Writing files to directory: "' + current_outdir + '"')

            # make sure the output directory exists
            if not os.path.exists(current_outdir):
                os.makedirs(current_outdir)
        else:
            current_outdir = outdir

        # Here is where the images are actually generated!
        try:
            file_writer = PngWriter(current_outdir)
            prefix = file_writer.unique_prefix()
            seeds = set()
            results = [] # list of filename, prompt pairs
            grid_images = dict() # seed -> Image, only used if `do_grid`
            def image_writer(image, seed, upscaled=False):
                if do_grid:
                    grid_images[seed] = image
                else:
                    if upscaled and opt.save_original:
                        filename = f'{prefix}.{seed}.postprocessed.png'
                    else:
                        filename = f'{prefix}.{seed}.png'
                    if opt.variation_amount > 0:
                        iter_opt = argparse.Namespace(**vars(opt)) # copy
                        this_variation = [[seed, opt.variation_amount]]
                        if opt.with_variations is None:
                            iter_opt.with_variations = this_variation
                        else:
                            iter_opt.with_variations = opt.with_variations + this_variation
                        iter_opt.variation_amount = 0
                        normalized_prompt = PromptFormatter(t2i, iter_opt).normalize_prompt()
                        metadata_prompt = f'{normalized_prompt} -S{iter_opt.seed}'
                    elif opt.with_variations is not None:
                        normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
                        metadata_prompt = f'{normalized_prompt} -S{opt.seed}' # use the original seed - the per-iteration value is the last variation-seed
                    else:
                        normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
                        metadata_prompt = f'{normalized_prompt} -S{seed}'
                    path = file_writer.save_image_and_prompt_to_png(image, metadata_prompt, filename)
                    if (not upscaled) or opt.save_original:
                        # only append to results if we didn't overwrite an earlier output
                        results.append([path, metadata_prompt])

                seeds.add(seed)

            t2i.prompt2image(image_callback=image_writer, **vars(opt))

            if do_grid and len(grid_images) > 0:
                grid_img = make_grid(list(grid_images.values()))
                first_seed = next(iter(seeds))
                filename = f'{prefix}.{first_seed}.png'
                # TODO better metadata for grid images
                normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
                metadata_prompt = f'{normalized_prompt} -S{first_seed} --grid -N{len(grid_images)}'
                path = file_writer.save_image_and_prompt_to_png(
                    grid_img, metadata_prompt, filename
                )
                results = [[path, metadata_prompt]]

            last_seeds = list(seeds)

        except AssertionError as e:
            print(e)
            continue

        except OSError as e:
            print(e)
            continue

        print('Outputs:')
        log_path = os.path.join(current_outdir, 'dream_log.txt')
        write_log_message(results, log_path)

    print('goodbye!')


def get_next_command(infile=None) -> str: #command string
    if infile is None:
        command = input('dream> ')
    else:
        command = infile.readline()
        if not command:
            raise EOFError
        else:
            command = command.strip()
        print(f'#{command}')
    return command

def dream_server_loop(t2i, host, port):
    print('\n* --web was specified, starting web server...')
    # Change working directory to the stable-diffusion directory
    os.chdir(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )

    # Start server
    DreamServer.model = t2i
    dream_server = ThreadingDreamServer((host, port))
    print(">> Started Stable Diffusion dream server!")
    if host == '0.0.0.0':
        print(f"Point your browser at http://localhost:{port} or use the host's DNS name or IP address.")
    else:
        print(">> Default host address now 127.0.0.1 (localhost). Use --host 0.0.0.0 to bind any address.")
        print(f">> Point your browser at http://{host}:{port}.")

    try:
        dream_server.serve_forever()
    except KeyboardInterrupt:
        pass

    dream_server.server_close()


def write_log_message(results, log_path):
    """logs the name of the output image, prompt, and prompt args to the terminal and log file"""
    log_lines = [f'{path}: {prompt}\n' for path, prompt in results]
    print(*log_lines, sep='')

    with open(log_path, 'a', encoding='utf-8') as file:
        file.writelines(log_lines)


SAMPLER_CHOICES=[
    'ddim',
    'k_dpm_2_a',
    'k_dpm_2',
    'k_euler_a',
    'k_euler',
    'k_heun',
    'k_lms',
    'plms',
]

def create_argv_parser():
    parser = argparse.ArgumentParser(
        description="""Generate images using Stable Diffusion.
        Use --web to launch the web interface. 
        Use --from_file to load prompts from a file path or standard input ("-").
        Otherwise you will be dropped into an interactive command prompt (type -h for help.)
        Other command-line arguments are defaults that can usually be overridden
        prompt the command prompt.
"""
    )
    parser.add_argument(
        '--laion400m',
        '--latent_diffusion',
        '-l',
        dest='laion400m',
        action='store_true',
        help='Fallback to the latent diffusion (laion400m) weights and config',
    )
    parser.add_argument(
        '--from_file',
        dest='infile',
        type=str,
        help='If specified, load prompts from this file',
    )
    parser.add_argument(
        '-n',
        '--iterations',
        type=int,
        default=1,
        help='Number of images to generate',
    )
    parser.add_argument(
        '-F',
        '--full_precision',
        dest='full_precision',
        action='store_true',
        help='Use more memory-intensive full precision math for calculations',
    )
    parser.add_argument(
        '-g',
        '--grid',
        action='store_true',
        help='Generate a grid instead of individual images',
    )
    parser.add_argument(
        '-A',
        '-m',
        '--sampler',
        dest='sampler_name',
        choices=SAMPLER_CHOICES,
        metavar='SAMPLER_NAME',
        default='k_lms',
        help=f'Set the initial sampler. Default: k_lms. Supported samplers: {", ".join(SAMPLER_CHOICES)}',
    )
    parser.add_argument(
        '--outdir',
        '-o',
        type=str,
        default='outputs/img-samples',
        help='Directory to save generated images and a log of prompts and seeds. Default: outputs/img-samples',
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        help='Path to a pre-trained embedding manager checkpoint - can only be set on command line',
    )
    parser.add_argument(
        '--prompt_as_dir',
        '-p',
        action='store_true',
        help='Place images in subdirectories named after the prompt.',
    )
    # GFPGAN related args
    parser.add_argument(
        '--gfpgan_bg_upsampler',
        type=str,
        default='realesrgan',
        help='Background upsampler. Default: realesrgan. Options: realesrgan, none. Only used if --gfpgan is specified',

    )
    parser.add_argument(
        '--gfpgan_bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400.',
    )
    parser.add_argument(
        '--gfpgan_model_path',
        type=str,
        default='experiments/pretrained_models/GFPGANv1.3.pth',
        help='Indicates the path to the GFPGAN model, relative to --gfpgan_dir.',
    )
    parser.add_argument(
        '--gfpgan_dir',
        type=str,
        default='../GFPGAN',
        help='Indicates the directory containing the GFPGAN code.',
    )
    parser.add_argument(
        '--web',
        dest='web',
        action='store_true',
        help='Start in web server mode.',
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Web server: Host or IP to listen on. Set to 0.0.0.0 to accept traffic from other devices on your network.'
    )
    parser.add_argument(
        '--port',
        type=int,
        default='9090',
        help='Web server: Port to listen on'
    )
    parser.add_argument(
        '--weights',
        default='model',
        help='Indicates the Stable Diffusion model to use.',
    )
    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help="device to run stable diffusion on. defaults to cuda `torch.cuda.current_device()` if available"
    )
    parser.add_argument(
        '--model',
        default='stable-diffusion-1.4',
        help='Indicates which diffusion model to load. (currently "stable-diffusion-1.4" (default) or "laion400m")',
    )
    parser.add_argument(
        '--config',
        default ='configs/models.yaml',
        help    ='Path to configuration file for alternate models.',
    )
    return parser


def create_cmd_parser():
    parser = argparse.ArgumentParser(
        description='Example: dream> a fantastic alien landscape -W1024 -H960 -s100 -n12'
    )
    parser.add_argument('prompt')
    parser.add_argument('-s', '--steps', type=int, help='Number of steps')
    parser.add_argument(
        '-S',
        '--seed',
        type=int,
        help='Image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc',
    )
    parser.add_argument(
        '-n',
        '--iterations',
        type=int,
        default=1,
        help='Number of samplings to perform (slower, but will provide seeds for individual images)',
    )
    parser.add_argument(
        '-W', '--width', type=int, help='Image width, multiple of 64'
    )
    parser.add_argument(
        '-H', '--height', type=int, help='Image height, multiple of 64'
    )
    parser.add_argument(
        '-C',
        '--cfg_scale',
        default=7.5,
        type=float,
        help='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder.',
    )
    parser.add_argument(
        '-g', '--grid', action='store_true', help='generate a grid'
    )
    parser.add_argument(
        '--outdir',
        '-o',
        type=str,
        default=None,
        help='Directory to save generated images and a log of prompts and seeds',
    )
    parser.add_argument(
        '-i',
        '--individual',
        action='store_true',
        help='Generate individual files (default)',
    )
    parser.add_argument(
        '-I',
        '--init_img',
        type=str,
        help='Path to input image for img2img mode (supersedes width and height)',
    )
    parser.add_argument(
        '-T',
        '-fit',
        '--fit',
        action='store_true',
        help='If specified, will resize the input image to fit within the dimensions of width x height (512x512 default)',
    )
    parser.add_argument(
        '-f',
        '--strength',
        default=0.75,
        type=float,
        help='Strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely',
    )
    parser.add_argument(
        '-G',
        '--gfpgan_strength',
        default=0,
        type=float,
        help='The strength at which to apply the GFPGAN model to the result, in order to improve faces.',
    )
    parser.add_argument(
        '-U',
        '--upscale',
        nargs='+',
        default=None,
        type=float,
        help='Scale factor (2, 4) for upscaling followed by upscaling strength (0-1.0). If strength not specified, defaults to 0.75'
    )
    parser.add_argument(
        '-save_orig',
        '--save_original',
        action='store_true',
        help='Save original. Use it when upscaling to save both versions.',
    )
    # variants is going to be superseded by a generalized "prompt-morph" function
    #    parser.add_argument('-v','--variants',type=int,help="in img2img mode, the first generated image will get passed back to img2img to generate the requested number of variants")
    parser.add_argument(
        '-x',
        '--skip_normalize',
        action='store_true',
        help='Skip subprompt weight normalization',
    )
    parser.add_argument(
        '-A',
        '-m',
        '--sampler',
        dest='sampler_name',
        default=None,
        type=str,
        choices=SAMPLER_CHOICES,
        metavar='SAMPLER_NAME',
        help=f'Switch to a different sampler. Supported samplers: {", ".join(SAMPLER_CHOICES)}',
    )
    parser.add_argument(
        '-t',
        '--log_tokenization',
        action='store_true',
        help='shows how the prompt is split into tokens'
    )
    parser.add_argument(
        '-v',
        '--variation_amount',
        default=0.0,
        type=float,
        help='If > 0, generates variations on the initial seed instead of random seeds per iteration. Must be between 0 and 1. Higher values will be more different.'
    )
    parser.add_argument(
        '-V',
        '--with_variations',
        default=None,
        type=str,
        help='list of variations to apply, in the format `seed:weight,seed:weight,...'
    )
    return parser


if __name__ == '__main__':
    main()
