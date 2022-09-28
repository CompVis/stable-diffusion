import os
import sys
import queue
from connect import StableDiffusionConnectionManager


if __name__ == "__main__":
    # get pid from command line
    pid = sys.argv[2]
    # clear sys.argv
    sys.argv = sys.argv[:1]
    StableDiffusionConnectionManager(pid=0)
