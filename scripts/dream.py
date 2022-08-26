#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import argparse
import shlex
import os
import sys
import copy
import warnings
import ldm.dream.readline
from ldm.dream.pngwriter import PngWriter, PromptFormatter

debugging = False


def main():
    """Initialize command-line parsers and the diffusion model"""
    arg_parser = create_argv_parser()
    opt = arg_parser.parse_args()
    if opt.laion400m:
        # defaults suitable to the older latent diffusion weights
        width = 256
        height = 256
        config = 'configs/latent-diffusion/txt2img-1p4B-eval.yaml'
        weights = 'models/ldm/text2img-large/model.ckpt'
    else:
        # some defaults suitable for stable diffusion weights
        width = 512
        height = 512
        config = 'configs/stable-diffusion/v1-inference.yaml'
        weights = 'models/ldm/stable-diffusion-v1/model.ckpt'

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
        latent_diffusion_weights=opt.laion400m,  # this is solely for recreating the prompt
        embedding_path=opt.embedding_path,
        device=opt.device,
    )

    # make sure the output directory exists
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    # gets rid of annoying messages about random seed
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    infile = None
    try:
        if opt.infile:
            infile = open(opt.infile, 'r')
    except FileNotFoundError as e:
        print(e)
        exit(-1)

    # preload the model
    t2i.load_model()

    # load GFPGAN if requested
    if opt.use_gfpgan:
        print('\n* --gfpgan was specified, loading gfpgan...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)

            try:
                model_path = os.path.join(
                    opt.gfpgan_dir, opt.gfpgan_model_path
                )
                if not os.path.isfile(model_path):
                    raise Exception(
                        'GFPGAN model not found at path ' + model_path
                    )

                sys.path.append(os.path.abspath(opt.gfpgan_dir))
                from gfpgan import GFPGANer

                bg_upsampler = load_gfpgan_bg_upsampler(
                    opt.gfpgan_bg_upsampler, opt.gfpgan_bg_tile
                )

                t2i.gfpgan = GFPGANer(
                    model_path=model_path,
                    upscale=opt.gfpgan_upscale,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=bg_upsampler,
                )
            except Exception:
                import traceback

                print('Error loading GFPGAN:', file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    print(
        "\n* Initialization done! Awaiting your command (-h for help, 'q' to quit, 'cd' to change output dir, 'pwd' to print output dir)..."
    )

    log_path = os.path.join(opt.outdir, 'dream_log.txt')
    with open(log_path, 'a') as log:
        cmd_parser = create_cmd_parser()
        main_loop(t2i, opt.outdir, cmd_parser, log, infile)
        log.close()
    if infile:
        infile.close()


def main_loop(t2i, outdir, parser, log, infile):
    """prompt/read/execute loop"""
    done = False
    last_seeds = []

    while not done:
        try:
            command = infile.readline() if infile else input('dream> ')
        except EOFError:
            done = True
            break

        if infile and len(command) == 0:
            done = True
            break

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

        if len(elements) == 0:
            continue

        if elements[0] == 'q':
            done = True
            break

        if elements[0] == 'cd' and len(elements) > 1:
            if os.path.exists(elements[1]):
                print(f'setting image output directory to {elements[1]}')
                outdir = elements[1]
            else:
                print(f'directory {elements[1]} does not exist')
            continue

        if elements[0] == 'pwd':
            print(f'current output directory is {outdir}')
            continue

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

        normalized_prompt = PromptFormatter(t2i, opt).normalize_prompt()
        individual_images = not opt.grid

        try:
            file_writer = PngWriter(outdir, normalized_prompt, opt.batch_size)
            callback = file_writer.write_image if individual_images else None

            image_list = t2i.prompt2image(image_callback=callback, **vars(opt))
            results = (
                file_writer.files_written if individual_images else image_list
            )

            if opt.grid and len(results) > 0:
                grid_img = file_writer.make_grid([r[0] for r in results])
                filename = file_writer.unique_filename(results[0][1])
                seeds = [a[1] for a in results]
                results = [[filename, seeds]]
                metadata_prompt = f'{normalized_prompt} -S{results[0][1]}'
                file_writer.save_image_and_prompt_to_png(
                    grid_img, metadata_prompt, filename
                )

            last_seeds = [r[1] for r in results]

        except AssertionError as e:
            print(e)
            continue

        except OSError as e:
            print(e)
            continue

        print('Outputs:')
        write_log_message(t2i, normalized_prompt, results, log)

    print('goodbye!')


def load_gfpgan_bg_upsampler(bg_upsampler, bg_tile=400):
    import torch

    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings

            warnings.warn(
                'The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                'If you really want to use it, please modify the corresponding codes.'
            )
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )  # need to set False in CPU mode
    else:
        bg_upsampler = None

    return bg_upsampler


# variant generation is going to be superseded by a generalized
# "prompt-morph" functionality
# def generate_variants(t2i,outdir,opt,previous_gens):
#     variants = []
#     print(f"Generating {opt.variants} variant(s)...")
#     newopt = copy.deepcopy(opt)
#     newopt.iterations = 1
#     newopt.variants   = None
#     for r in previous_gens:
#         newopt.init_img = r[0]
#         prompt            = PromptFormatter(t2i,newopt).normalize_prompt()
#         print(f"] generating variant for {newopt.init_img}")
#         for j in range(0,opt.variants):
#             try:
#                 file_writer        = PngWriter(outdir,prompt,newopt.batch_size)
#                 callback           = file_writer.write_image
#                 t2i.prompt2image(image_callback=callback,**vars(newopt))
#                 results           = file_writer.files_written
#                 variants.append([prompt,results])
#             except AssertionError as e:
#                 print(e)
#                 continue
#     print(f'{opt.variants} variants generated')
#     return variants


def write_log_message(t2i, prompt, results, logfile):
    """logs the name of the output image, its prompt and seed to the terminal, log file, and a Dream text chunk in the PNG metadata"""
    last_seed = None
    img_num = 1
    seenit = {}

    for r in results:
        seed = r[1]
        log_message = f'{r[0]}: {prompt} -S{seed}'

        print(log_message)
        logfile.write(log_message + '\n')
        logfile.flush()


def create_argv_parser():
    parser = argparse.ArgumentParser(
        description="Parse script's command line args"
    )
    parser.add_argument(
        '--laion400m',
        '--latent_diffusion',
        '-l',
        dest='laion400m',
        action='store_true',
        help='fallback to the latent diffusion (laion400m) weights and config',
    )
    parser.add_argument(
        '--from_file',
        dest='infile',
        type=str,
        help='if specified, load prompts from this file',
    )
    parser.add_argument(
        '-n',
        '--iterations',
        type=int,
        default=1,
        help='number of images to generate',
    )
    parser.add_argument(
        '-F',
        '--full_precision',
        dest='full_precision',
        action='store_true',
        help='use slower full precision math for calculations',
    )
    parser.add_argument(
        '--sampler',
        '-m',
        dest='sampler_name',
        choices=[
            'ddim',
            'k_dpm_2_a',
            'k_dpm_2',
            'k_euler_a',
            'k_euler',
            'k_heun',
            'k_lms',
            'plms',
        ],
        default='k_lms',
        help='which sampler to use (k_lms) - can only be set on command line',
    )
    parser.add_argument(
        '--outdir',
        '-o',
        type=str,
        default='outputs/img-samples',
        help='directory in which to place generated images and a log of prompts and seeds (outputs/img-samples',
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        help='Path to a pre-trained embedding manager checkpoint - can only be set on command line',
    )
    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to run stable diffusion on. defaults to cuda `torch.cuda.current_device()` if avalible',
    )
    # GFPGAN related args
    parser.add_argument(
        '--gfpgan',
        dest='use_gfpgan',
        action='store_true',
        help='load gfpgan for use in the dreambot. Note: Enabling GFPGAN will require more GPU memory',
    )
    parser.add_argument(
        '--gfpgan_upscale',
        type=int,
        default=2,
        help='The final upsampling scale of the image. Default: 2. Only used if --gfpgan is specified',
    )
    parser.add_argument(
        '--gfpgan_bg_upsampler',
        type=str,
        default='realesrgan',
        help='Background upsampler. Default: None. Options: realesrgan, none. Only used if --gfpgan is specified',
    )
    parser.add_argument(
        '--gfpgan_bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400. Only used if --gfpgan is specified',
    )
    parser.add_argument(
        '--gfpgan_model_path',
        type=str,
        default='experiments/pretrained_models/GFPGANv1.3.pth',
        help='indicates the path to the GFPGAN model, relative to --gfpgan_dir. Only used if --gfpgan is specified',
    )
    parser.add_argument(
        '--gfpgan_dir',
        type=str,
        default='../GFPGAN',
        help='indicates the directory containing the GFPGAN code. Only used if --gfpgan is specified',
    )
    return parser


def create_cmd_parser():
    parser = argparse.ArgumentParser(
        description='Example: dream> a fantastic alien landscape -W1024 -H960 -s100 -n12'
    )
    parser.add_argument('prompt')
    parser.add_argument('-s', '--steps', type=int, help='number of steps')
    parser.add_argument(
        '-S',
        '--seed',
        type=int,
        help='image seed; a +ve integer, or use -1 for the previous seed, -2 for the one before that, etc',
    )
    parser.add_argument(
        '-n',
        '--iterations',
        type=int,
        default=1,
        help='number of samplings to perform (slower, but will provide seeds for individual images)',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=1,
        help='number of images to produce per sampling (will not provide seeds for individual images!)',
    )
    parser.add_argument(
        '-W', '--width', type=int, help='image width, multiple of 64'
    )
    parser.add_argument(
        '-H', '--height', type=int, help='image height, multiple of 64'
    )
    parser.add_argument(
        '-C',
        '--cfg_scale',
        default=7.5,
        type=float,
        help='prompt configuration scale',
    )
    parser.add_argument(
        '-g', '--grid', action='store_true', help='generate a grid'
    )
    parser.add_argument(
        '-i',
        '--individual',
        action='store_true',
        help='generate individual files (default)',
    )
    parser.add_argument(
        '-I',
        '--init_img',
        type=str,
        help='path to input image for img2img mode (supersedes width and height)',
    )
    parser.add_argument(
        '-f',
        '--strength',
        default=0.75,
        type=float,
        help='strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely',
    )
    parser.add_argument(
        '-G',
        '--gfpgan_strength',
        default=0.5,
        type=float,
        help='The strength at which to apply the GFPGAN model to the result, in order to improve faces.',
    )
    # variants is going to be superseded by a generalized "prompt-morph" function
    #    parser.add_argument('-v','--variants',type=int,help="in img2img mode, the first generated image will get passed back to img2img to generate the requested number of variants")
    parser.add_argument(
        '-x',
        '--skip_normalize',
        action='store_true',
        help='skip subprompt weight normalization',
    )
    return parser


if __name__ == '__main__':
    main()
