from modules.parse_seed_weights import parse_seed_weights
import argparse

SAMPLER_CHOICES = [
    "ddim",
    "k_dpm_2_a",
    "k_dpm_2",
    "k_euler_a",
    "k_euler",
    "k_heun",
    "k_lms",
    "plms",
]


def parameters_to_command(params):
    """
    Converts dict of parameters into a `dream.py` REPL command.
    """

    switches = list()

    if "prompt" in params:
        switches.append(f'"{params["prompt"]}"')
    if "steps" in params:
        switches.append(f'-s {params["steps"]}')
    if "seed" in params:
        switches.append(f'-S {params["seed"]}')
    if "width" in params:
        switches.append(f'-W {params["width"]}')
    if "height" in params:
        switches.append(f'-H {params["height"]}')
    if "cfg_scale" in params:
        switches.append(f'-C {params["cfg_scale"]}')
    if "sampler_name" in params:
        switches.append(f'-A {params["sampler_name"]}')
    if "seamless" in params and params["seamless"] == True:
        switches.append(f"--seamless")
    if "init_img" in params and len(params["init_img"]) > 0:
        switches.append(f'-I {params["init_img"]}')
    if "init_mask" in params and len(params["init_mask"]) > 0:
        switches.append(f'-M {params["init_mask"]}')
    if "init_color" in params and len(params["init_color"]) > 0:
        switches.append(f'--init_color {params["init_color"]}')
    if "strength" in params and "init_img" in params:
        switches.append(f'-f {params["strength"]}')
        if "fit" in params and params["fit"] == True:
            switches.append(f"--fit")
    if "gfpgan_strength" in params and params["gfpgan_strength"]:
        switches.append(f'-G {params["gfpgan_strength"]}')
    if "upscale" in params and params["upscale"]:
        switches.append(f'-U {params["upscale"][0]} {params["upscale"][1]}')
    if "variation_amount" in params and params["variation_amount"] > 0:
        switches.append(f'-v {params["variation_amount"]}')
        if "with_variations" in params:
            seed_weight_pairs = ",".join(
                f"{seed}:{weight}" for seed, weight in params["with_variations"]
            )
            switches.append(f"-V {seed_weight_pairs}")

    return " ".join(switches)
