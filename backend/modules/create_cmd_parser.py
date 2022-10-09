import argparse
import os
from ldm.invoke.args import PRECISION_CHOICES


def create_cmd_parser():
    parser = argparse.ArgumentParser(description="InvokeAI web UI")
    parser.add_argument(
        "--host",
        type=str,
        help="The host to serve on",
        default="localhost",
    )
    parser.add_argument("--port", type=int, help="The port to serve on", default=9090)
    parser.add_argument(
        "--cors",
        nargs="*",
        type=str,
        help="Additional allowed origins, comma-separated",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        help="Path to a pre-trained embedding manager checkpoint - can only be set on command line",
    )
    # TODO: Can't get flask to serve images from any dir (saving to the dir does work when specified)
    # parser.add_argument(
    #     "--output_dir",
    #     default="outputs/",
    #     type=str,
    #     help="Directory for output images",
    # )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enables verbose logging",
    )
    parser.add_argument(
        "--precision",
        dest="precision",
        type=str,
        choices=PRECISION_CHOICES,
        metavar="PRECISION",
        help=f'Set model precision. Defaults to auto selected based on device. Options: {", ".join(PRECISION_CHOICES)}',
        default="auto",
    )
    parser.add_argument(
        '--free_gpu_mem',
        dest='free_gpu_mem',
        action='store_true',
        help='Force free gpu memory before final decoding',
    )

    return parser
