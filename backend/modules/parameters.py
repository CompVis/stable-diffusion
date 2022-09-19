from modules.parse_seed_weights import parse_seed_weights
import argparse

SAMPLER_CHOICES = [
    'ddim',
    'k_dpm_2_a',
    'k_dpm_2',
    'k_euler_a',
    'k_euler',
    'k_heun',
    'k_lms',
    'plms',
]


def parameters_to_command(params):
    """
    Converts dict of parameters into a `dream.py` REPL command.
    """

    switches = list()

    if 'prompt' in params:
        switches.append(f'"{params["prompt"]}"')
    if 'steps' in params:
        switches.append(f'-s {params["steps"]}')
    if 'seed' in params:
        switches.append(f'-S {params["seed"]}')
    if 'width' in params:
        switches.append(f'-W {params["width"]}')
    if 'height' in params:
        switches.append(f'-H {params["height"]}')
    if 'cfg_scale' in params:
        switches.append(f'-C {params["cfg_scale"]}')
    if 'sampler_name' in params:
        switches.append(f'-A {params["sampler_name"]}')
    if 'seamless' in params and params["seamless"] == True:
        switches.append(f'--seamless')
    if 'init_img' in params and len(params['init_img']) > 0:
        switches.append(f'-I {params["init_img"]}')
    if 'init_mask' in params and len(params['init_mask']) > 0:
        switches.append(f'-M {params["init_mask"]}')
    if 'init_color' in params and len(params['init_color']) > 0:
        switches.append(f'--init_color {params["init_color"]}')
    if 'strength' in params and 'init_img' in params:
        switches.append(f'-f {params["strength"]}')
        if 'fit' in params and params["fit"] == True:
            switches.append(f'--fit')
    if 'gfpgan_strength' in params and params["gfpgan_strength"]:
        switches.append(f'-G {params["gfpgan_strength"]}')
    if 'upscale' in params and params["upscale"]:
        switches.append(f'-U {params["upscale"][0]} {params["upscale"][1]}')
    if 'variation_amount' in params and params['variation_amount'] > 0:
        switches.append(f'-v {params["variation_amount"]}')
        if 'with_variations' in params:
            seed_weight_pairs = ','.join(f'{seed}:{weight}' for seed, weight in params["with_variations"])
            switches.append(f'-V {seed_weight_pairs}')

    return ' '.join(switches)



def create_cmd_parser():
    """
    This is simply a copy of the parser from `dream.py` with a change to give
    prompt a default value. This is a temporary hack pending merge of #587 which
    provides a better way to do this.
    """
    parser = argparse.ArgumentParser(
        description='Example: dream> a fantastic alien landscape -W1024 -H960 -s100 -n12',
        exit_on_error=True,
    )
    parser.add_argument('prompt', nargs='?', default='')
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
        '--seamless',
        action='store_true',
        help='Change the model to seamless tiling (circular) mode',
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
        '-M',
        '--init_mask',
        type=str,
        help='Path to input mask for inpainting mode (supersedes width and height)',
    )
    parser.add_argument(
        '--init_color',
        type=str,
        help='Path to reference image for color correction (used for repeated img2img and inpainting)'
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
