#!/usr/bin/env python

import readline
import argparse
import shlex
import atexit
from os import path

def main():
    arg_parser = create_argv_parser()
    opt        = arg_parser.parse_args()
    if opt.laion400m:
        # defaults suitable to the older latent diffusion weights
        width   = 256
        height  = 256
        config  = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        weights = "models/ldm/text2img-large/model.ckpt"
    else:
        # some defaults suitable for stable diffusion weights
        width   = 512
        height  = 512
        config  = "configs/stable-diffusion/v1-inference.yaml"
        weights = "models/ldm/stable-diffusion-v1/model.ckpt"

    # command line history will be stored in a file called "~/.dream_history"
    load_history()

    print("* Initializing, be patient...\n")
    from pytorch_lightning import logging
    from ldm.simplet2i import T2I

    # creating a simple text2image object with a handful of
    # defaults passed on the command line.
    # additional parameters will be added (or overriden) during
    # the user input loop
    t2i = T2I(width=width,
              height=height,
              batch=opt.batch,
              outdir=opt.outdir,
              sampler=opt.sampler,
              weights=weights,
              config=config)

    # gets rid of annoying messages about random seed
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # preload the model
    t2i.load_model()
    print("\n* Initialization done! Awaiting your command...")

    log_path   = path.join(opt.outdir,"dream_log.txt")
    with open(log_path,'a') as log:
        cmd_parser = create_cmd_parser()
        main_loop(t2i,cmd_parser,log)
        log.close()

def main_loop(t2i,parser,log):
    while True:
        try:
            command = input("dream> ")
        except EOFError:
            print("goodbye!")
            break

        elements = shlex.split(command)
        switches = ['']
        switches_started = False

        for el in elements:
            if el[0]=='-' and not switches_started:
                switches_started = True
            if switches_started:
                switches.append(el)
            else:
                switches[0] += el
                switches[0] += ' '
        switches[0] = switches[0][:len(switches[0])-1]
        try:
            opt      = parser.parse_args(switches)
        except SystemExit:
            parser.print_help()
            pass
        results = t2i.txt2img(**vars(opt))
        print("Outputs:")
        for r in results:
            log_message = " ".join(['   ',str(r[0])+':',
                                    f'"{switches[0]}"',
                                    *switches[1:],f'-S {r[1]}'])
            print(log_message)
            log.write(log_message+"\n")
            log.flush()

def create_argv_parser():
    parser = argparse.ArgumentParser(description="Parse script's command line args")
    parser.add_argument("--laion400m",
                        "--latent_diffusion",
                        "-l",
                        dest='laion400m',
                        action='store_true',
                        help="fallback to the latent diffusion (LAION4400M) weights and config")
    parser.add_argument('-n','--iterations',
                        type=int,
                        default=1,
                        help="number of images to produce per sampling (overrides -n<iterations>, faster but doesn't produce individual seeds)")
    parser.add_argument('-b','--batch',
                        type=int,
                        default=1,
                        help="number of images to produce per sampling (currently broken")
    parser.add_argument('--sampler',
                        choices=['plms','ddim'],
                        default='plms',
                        help="which sampler to use")
    parser.add_argument('-o',
                        '--outdir',
                        type=str,
                        default="outputs/txt2img-samples",
                        help="directory in which to place generated images and a log of prompts and seeds")
    return parser
                        
    
def create_cmd_parser():
    parser = argparse.ArgumentParser(description="Parse terminal input in a discord 'dreambot' fashion")
    parser.add_argument('prompt')
    parser.add_argument('-s','--steps',type=int,help="number of steps")
    parser.add_argument('-S','--seed',type=int,help="image seed")
    parser.add_argument('-n','--iterations',type=int,default=1,help="number of samplings to perform")
    parser.add_argument('-b','--batch',type=int,default=1,help="number of images to produce per sampling (currently broken)")
    parser.add_argument('-W','--width',type=int,help="image width, multiple of 64")
    parser.add_argument('-H','--height',type=int,help="image height, multiple of 64")
    parser.add_argument('-C','--cfg_scale',type=float,help="prompt configuration scale (7.5)")
    parser.add_argument('-g','--grid',action='store_true',help="generate a grid")
    return parser

def load_history():
    histfile = path.join(path.expanduser('~'),".dream_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file,histfile)

if __name__ == "__main__":
    main()
