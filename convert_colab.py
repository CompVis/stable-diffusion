from colab_convert import convert
from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument(
    "-o",
    "--output",
    default="./Deforum_Stable_Diffusion_test.ipynb",
    help="Output ipynb colab file",
)

convert(
    "./Deforum_Stable_Diffusion.py",
    argp.parse_args().output,
    extra_flags={},
)