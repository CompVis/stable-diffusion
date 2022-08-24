#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import argparse
import shlex
import atexit
import os
import sys
from PIL import Image,PngImagePlugin

# readline unavailable on windows systems
try:
    import readline
    readline_available = True
except:
    readline_available = False

debugging = False

def main():
    ''' Initialize command-line parsers and the diffusion model '''
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
    if readline_available:
        setup_readline()

    print("* Initializing, be patient...\n")
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
    t2i = T2I(width=width,
              height=height,
              batch_size=opt.batch_size,
              outdir=opt.outdir,
              sampler_name=opt.sampler_name,
              weights=weights,
              full_precision=opt.full_precision,
              config=config,
              latent_diffusion_weights=opt.laion400m # this is solely for recreating the prompt
    )

    # make sure the output directory exists
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
        
    # gets rid of annoying messages about random seed
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    infile = None
    try:
        if opt.infile is not None:
            infile = open(opt.infile,'r')
    except FileNotFoundError as e:
        print(e)
        exit(-1)

    # preload the model
    if not debugging:
        t2i.load_model()
    print("\n* Initialization done! Awaiting your command (-h for help, 'q' to quit, 'cd' to change output dir, 'pwd' to print output dir)...")

    log_path   = os.path.join(opt.outdir,'dream_log.txt')
    with open(log_path,'a') as log:
        cmd_parser = create_cmd_parser()
        main_loop(t2i,cmd_parser,log,infile)
        log.close()
    if infile:
        infile.close()


def main_loop(t2i,parser,log,infile):
    ''' prompt/read/execute loop '''
    done = False
    
    while not done:
        try:
            command = infile.readline() if infile else input("dream> ") 
        except EOFError:
            done = True
            break

        if infile and len(command)==0:
            done = True
            break

        if command.startswith(('#','//')):
            continue

        # before splitting, escape single quotes so as not to mess
        # up the parser
        command = command.replace("'","\\'")

        try:
            elements = shlex.split(command)
        except ValueError as e:
            print(str(e))
            continue
        
        if len(elements)==0:
            continue

        if elements[0]=='q':
            done = True
            break

        if elements[0]=='cd' and len(elements)>1:
            if os.path.exists(elements[1]):
                print(f"setting image output directory to {elements[1]}")
                t2i.outdir=elements[1]
            else:
                print(f"directory {elements[1]} does not exist")
            continue

        if elements[0]=='pwd':
            print(f"current output directory is {t2i.outdir}")
            continue
        
        if elements[0].startswith('!dream'): # in case a stored prompt still contains the !dream command
            elements.pop(0)
            
        # rearrange the arguments to mimic how it works in the Dream bot.
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
            continue
        if len(opt.prompt)==0:
            print("Try again with a prompt!")
            continue

        try:
            if opt.init_img is None:
                results = t2i.txt2img(**vars(opt))
            else:
                results = t2i.img2img(**vars(opt))
        except AssertionError as e:
            print(e)
            continue
        print("Outputs:")
        write_log_message(t2i,opt,results,log)
            

    print("goodbye!")


def write_log_message(t2i,opt,results,logfile):
    ''' logs the name of the output image, its prompt and seed to the terminal, log file, and a Dream text chunk in the PNG metadata '''
    switches = _reconstruct_switches(t2i,opt)
    prompt_str = ' '.join(switches)

    # when multiple images are produced in batch, then we keep track of where each starts
    last_seed  = None
    img_num    = 1
    batch_size = opt.batch_size or t2i.batch_size
    seenit     = {}

    seeds = [a[1] for a in results]
    if batch_size > 1:
        seeds = f"(seeds for each batch row: {seeds})"
    else:
        seeds = f"(seeds for individual images: {seeds})"

    for r in results:
        seed = r[1]
        log_message = (f'{r[0]}: {prompt_str} -S{seed}')

        if batch_size > 1:
            if seed != last_seed:
                img_num = 1
                log_message += f' # (batch image {img_num} of {batch_size})'
            else:
                img_num += 1
                log_message += f' # (batch image {img_num} of {batch_size})'
            last_seed = seed
        print(log_message)
        logfile.write(log_message+"\n")
        logfile.flush()
        if r[0] not in seenit:
            seenit[r[0]] = True
            try:
                if opt.grid:
                    _write_prompt_to_png(r[0],f'{prompt_str} -g -S{seed} {seeds}')
                else:
                    _write_prompt_to_png(r[0],f'{prompt_str} -S{seed}')
            except FileNotFoundError:
                print(f"Could not open file '{r[0]}' for reading")

def _reconstruct_switches(t2i,opt):
    '''Normalize the prompt and switches'''
    switches = list()
    switches.append(f'"{opt.prompt}"')
    switches.append(f'-s{opt.steps        or t2i.steps}')
    switches.append(f'-b{opt.batch_size   or t2i.batch_size}')
    switches.append(f'-W{opt.width        or t2i.width}')
    switches.append(f'-H{opt.height       or t2i.height}')
    switches.append(f'-C{opt.cfg_scale    or t2i.cfg_scale}')
    switches.append(f'-m{t2i.sampler_name}')
    if opt.init_img:
        switches.append(f'-I{opt.init_img}')
    if opt.strength and opt.init_img is not None:
        switches.append(f'-f{opt.strength or t2i.strength}')
    if t2i.full_precision:
        switches.append('-F')
    return switches

def _write_prompt_to_png(path,prompt):
    info = PngImagePlugin.PngInfo()
    info.add_text("Dream",prompt)
    im = Image.open(path)
    im.save(path,"PNG",pnginfo=info)
    
def create_argv_parser():
    parser = argparse.ArgumentParser(description="Parse script's command line args")
    parser.add_argument("--laion400m",
                        "--latent_diffusion",
                        "-l",
                        dest='laion400m',
                        action='store_true',
                        help="fallback to the latent diffusion (laion400m) weights and config")
    parser.add_argument("--from_file",
                        dest='infile',
                        type=str,
                        help="if specified, load prompts from this file")
    parser.add_argument('-n','--iterations',
                        type=int,
                        default=1,
                        help="number of images to generate")
    parser.add_argument('-F','--full_precision',
                        dest='full_precision',
                        action='store_true',
                        help="use slower full precision math for calculations")
    parser.add_argument('-b','--batch_size',
                        type=int,
                        default=1,
                        help="number of images to produce per iteration (faster, but doesn't generate individual seeds")
    parser.add_argument('--sampler','-m',
                        dest="sampler_name",
                        choices=['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms'],
                        default='k_lms',
                        help="which sampler to use (k_lms) - can only be set on command line")
    parser.add_argument('--outdir',
                        '-o',
                        type=str,
                        default="outputs/img-samples",
                        help="directory in which to place generated images and a log of prompts and seeds")
    return parser
                        
    
def create_cmd_parser():
    parser = argparse.ArgumentParser(description='Example: dream> a fantastic alien landscape -W1024 -H960 -s100 -n12')
    parser.add_argument('prompt')
    parser.add_argument('-s','--steps',type=int,help="number of steps")
    parser.add_argument('-S','--seed',type=int,help="image seed")
    parser.add_argument('-n','--iterations',type=int,default=1,help="number of samplings to perform (slower, but will provide seeds for individual images)")
    parser.add_argument('-b','--batch_size',type=int,default=1,help="number of images to produce per sampling (will not provide seeds for individual images!)")
    parser.add_argument('-W','--width',type=int,help="image width, multiple of 64")
    parser.add_argument('-H','--height',type=int,help="image height, multiple of 64")
    parser.add_argument('-C','--cfg_scale',default=7.5,type=float,help="prompt configuration scale")
    parser.add_argument('-g','--grid',action='store_true',help="generate a grid")
    parser.add_argument('-i','--individual',action='store_true',help="generate individual files (default)")
    parser.add_argument('-I','--init_img',type=str,help="path to input image (supersedes width and height)")
    parser.add_argument('-f','--strength',default=0.75,type=float,help="strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely")
    parser.add_argument('-x','--skip_normalize',action='store_true',help="skip subprompt weight normalization")
    return parser

if readline_available:
    def setup_readline():
        readline.set_completer(Completer(['cd','pwd',
                                          '--steps','-s','--seed','-S','--iterations','-n','--batch_size','-b',
                                          '--width','-W','--height','-H','--cfg_scale','-C','--grid','-g',
                                          '--individual','-i','--init_img','-I','--strength','-f']).complete)
        readline.set_completer_delims(" ")
        readline.parse_and_bind('tab: complete')
        load_history()

    def load_history():
        histfile = os.path.join(os.path.expanduser('~'),".dream_history")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file,histfile)

    class Completer():
        def __init__(self,options):
            self.options = sorted(options)
            return

        def complete(self,text,state):
            buffer = readline.get_line_buffer()
            
            if text.startswith(('-I','--init_img')):
                return self._path_completions(text,state,('.png'))

            if buffer.strip().endswith('cd') or text.startswith(('.','/')):
                return self._path_completions(text,state,())

            response = None
            if state == 0:
                # This is the first time for this text, so build a match list.
                if text:
                    self.matches = [s 
                                    for s in self.options
                                    if s and s.startswith(text)]
                else:
                    self.matches = self.options[:]

            # Return the state'th item from the match list,
            # if we have that many.
            try:
                response = self.matches[state]
            except IndexError:
                response = None
            return response

        def _path_completions(self,text,state,extensions):
            # get the path so far
            if text.startswith('-I'):
                path = text.replace('-I','',1).lstrip()
            elif text.startswith('--init_img='):
                path = text.replace('--init_img=','',1).lstrip()
            else:
                path = text

            matches  = list()

            path = os.path.expanduser(path)
            if len(path)==0:
                matches.append(text+'./')
            else:
                dir  = os.path.dirname(path)
                dir_list = os.listdir(dir)
                for n in dir_list:
                    if n.startswith('.') and len(n)>1:
                        continue
                    full_path = os.path.join(dir,n)
                    if full_path.startswith(path):
                        if os.path.isdir(full_path):
                            matches.append(os.path.join(os.path.dirname(text),n)+'/')
                        elif n.endswith(extensions):
                            matches.append(os.path.join(os.path.dirname(text),n))

            try:
                response = matches[state]
            except IndexError:
                response = None
            return response

if __name__ == "__main__":
    main()

