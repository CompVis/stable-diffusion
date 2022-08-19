#!/usr/bin/env python
import argparse
import shlex
import atexit
import os

# readline unavailable on windows systems
try:
    import readline
    readline_available = True
except:
    readline_available = False

debugging = True

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
    from pytorch_lightning import logging
    from ldm.simplet2i import T2I

    # creating a simple text2image object with a handful of
    # defaults passed on the command line.
    # additional parameters will be added (or overriden) during
    # the user input loop
    t2i = T2I(width=width,
              height=height,
              batch_size=opt.batch_size,
              outdir=opt.outdir,
              sampler=opt.sampler,
              weights=weights,
              config=config)

    # make sure the output directory exists
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
        
    # gets rid of annoying messages about random seed
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # preload the model
    if not debugging:
        t2i.load_model()
    print("\n* Initialization done! Awaiting your command (-h for help, q to quit)...")

    log_path   = os.path.join(opt.outdir,"dream_log.txt")
    with open(log_path,'a') as log:
        cmd_parser = create_cmd_parser()
        main_loop(t2i,cmd_parser,log)
        log.close()

def main_loop(t2i,parser,log):
    ''' prompt/read/execute loop '''
    done = False
    
    while not done:
        try:
            command = input("dream> ")
        except EOFError:
            done = True
            break

        elements = shlex.split(command)
        if elements[0]=='q':  # 
            done = True
            break
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
            print("Outputs:")
            write_log_message(opt,switches,results,log)
        except KeyboardInterrupt:
            print('*interrupted*')
            continue
    print("goodbye!")


def write_log_message(opt,switches,results,logfile):
    ''' logs the name of the output image, its prompt and seed to both the terminal and the log file '''
    if opt.grid:
        _output_for_grid(switches,results,logfile)
    else:
        _output_for_individual(switches,results,logfile)

def _output_for_individual(switches,results,logfile):
    for r in results:
        log_message = " ".join(['   ',str(r[0])+':',
                                f'"{switches[0]}"',
                                *switches[1:],f'-S {r[1]}'])
        print(log_message)
        logfile.write(log_message+"\n")
        logfile.flush()

def _output_for_grid(switches,results,logfile):
    first_seed = results[0][1]
    log_message = " ".join(['   ',str(results[0][0])+':',
                            f'"{switches[0]}"',
                            *switches[1:],f'-S {results[0][1]}'])
    print(log_message)
    logfile.write(log_message+"\n")
    all_seeds   = [row[1] for row in results]
    log_message = f'    seeds for individual rows: {all_seeds}'
    print(log_message)
    logfile.write(log_message+"\n")

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
                        help="number of images to generate")
    parser.add_argument('-b','--batch_size',
                        type=int,
                        default=1,
                        help="number of images to produce per iteration (currently not working properly - producing too many images)")
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
    parser = argparse.ArgumentParser(description='Example: dream> a fantastic alien landscape -W1024 -H960 -s100 -n12')
    parser.add_argument('prompt')
    parser.add_argument('-s','--steps',type=int,help="number of steps")
    parser.add_argument('-S','--seed',type=int,help="image seed")
    parser.add_argument('-n','--iterations',type=int,default=1,help="number of samplings to perform")
    parser.add_argument('-b','--batch_size',type=int,default=1,help="number of images to produce per sampling (currently broken)")
    parser.add_argument('-W','--width',type=int,help="image width, multiple of 64")
    parser.add_argument('-H','--height',type=int,help="image height, multiple of 64")
    parser.add_argument('-C','--cfg_scale',default=7.5,type=float,help="prompt configuration scale")
    parser.add_argument('-g','--grid',action='store_true',help="generate a grid")
    parser.add_argument('-i','--individual',action='store_true',help="generate individual files (default)")
    parser.add_argument('-I','--init_img',type=str,help="path to input image (supersedes width and height)")
    parser.add_argument('-f','--strength',default=0.75,type=float,help="strength for noising/unnoising. 0.0 preserves image exactly, 1.0 replaces it completely")
    return parser

if readline_available:
    def setup_readline():
        readline.set_completer(Completer(['--steps','-s','--seed','-S','--iterations','-n','--batch_size','-b',
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
            if text.startswith('-I') or text.startswith('--init_img'):
                return self._image_completions(text,state)

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

        def _image_completions(self,text,state):
            # get the path so far
            if text.startswith('-I'):
                path = text.replace('-I','',1).lstrip()
            elif text.startswith('--init_img='):
                path = text.replace('--init_img=','',1).lstrip()

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
                        elif n.endswith('.png'):
                            matches.append(os.path.join(os.path.dirname(text),n))

            try:
                response = matches[state]
            except IndexError:
                response = None
            return response
        

if __name__ == "__main__":
    main()
