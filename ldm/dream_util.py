'''Utilities for dealing with PNG images and their path names'''
import os
import atexit
import re
from math import sqrt,floor,ceil
from PIL import Image,PngImagePlugin

# -------------------image generation utils-----
class PngWriter:

    def __init__(self,outdir,prompt=None,batch_size=1):
        self.outdir           = outdir
        self.batch_size       = batch_size
        self.prompt           = prompt
        self.filepath         = None
        self.files_written    = []
        os.makedirs(outdir, exist_ok=True)

    def write_image(self,image,seed):
        self.filepath = self.unique_filename(seed,self.filepath) # will increment name in some sensible way
        try:
            prompt = f'{self.prompt} -S{seed}'
            self.save_image_and_prompt_to_png(image,prompt,self.filepath)
        except IOError as e:
            print(e)
        self.files_written.append([self.filepath,seed])

    def unique_filename(self,seed,previouspath=None):
        revision = 1

        if previouspath is None:
            # sort reverse alphabetically until we find max+1
            dirlist   = sorted(os.listdir(self.outdir),reverse=True)
            # find the first filename that matches our pattern or return 000000.0.png
            filename   = next((f for f in dirlist if re.match('^(\d+)\..*\.png',f)),'0000000.0.png')
            basecount  = int(filename.split('.',1)[0])
            basecount += 1
            if self.batch_size > 1:
                filename = f'{basecount:06}.{seed}.01.png'
            else:
                filename = f'{basecount:06}.{seed}.png'
            return os.path.join(self.outdir,filename)

        else:
            basename = os.path.basename(previouspath)
            x = re.match('^(\d+)\..*\.png',basename)
            if not x:
                return self.unique_filename(seed,previouspath)

            basecount = int(x.groups()[0])
            series = 0 
            finished = False
            while not finished:
                series += 1
                filename = f'{basecount:06}.{seed}.png'
                if self.batch_size>1 or os.path.exists(os.path.join(self.outdir,filename)):
                    filename = f'{basecount:06}.{seed}.{series:02}.png'
                finished = not os.path.exists(os.path.join(self.outdir,filename))
            return os.path.join(self.outdir,filename)

    def save_image_and_prompt_to_png(self,image,prompt,path):
        info = PngImagePlugin.PngInfo()
        info.add_text("Dream",prompt)
        image.save(path,"PNG",pnginfo=info)

    def make_grid(self,image_list,rows=None,cols=None):
        image_cnt = len(image_list)
        if None in (rows,cols):
            rows = floor(sqrt(image_cnt))  # try to make it square
            cols = ceil(image_cnt/rows)
        width  = image_list[0].width
        height = image_list[0].height

        grid_img = Image.new('RGB',(width*cols,height*rows))
        for r in range(0,rows):
            for c in range (0,cols):
                i = r*rows + c
                grid_img.paste(image_list[i],(c*width,r*height))

        return grid_img
    
class PromptFormatter():
    def __init__(self,t2i,opt):
        self.t2i = t2i
        self.opt = opt

    def normalize_prompt(self):
        '''Normalize the prompt and switches'''
        t2i      = self.t2i
        opt      = self.opt

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
        return ' '.join(switches)

# ---------------readline utilities---------------------
try:
    import readline
    readline_available = True
except:
    readline_available = False

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

if readline_available:
    readline.set_completer(Completer(['cd','pwd',
                                      '--steps','-s','--seed','-S','--iterations','-n','--batch_size','-b',
                                      '--width','-W','--height','-H','--cfg_scale','-C','--grid','-g',
                                      '--individual','-i','--init_img','-I','--strength','-f','-v','--variants']).complete)
    readline.set_completer_delims(" ")
    readline.parse_and_bind('tab: complete')

    histfile = os.path.join(os.path.expanduser('~'),".dream_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file,histfile)

