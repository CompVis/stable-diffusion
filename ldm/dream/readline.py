"""
Readline helper functions for dream.py (linux and mac only).
"""
import os
import re
import atexit

# ---------------readline utilities---------------------
try:
    import readline

    readline_available = True
except:
    readline_available = False


class Completer:
    def __init__(self, options):
        self.options = sorted(options)
        return

    def complete(self, text, state):
        buffer = readline.get_line_buffer()

        if text.startswith(('-I', '--init_img','-M','--init_mask')):
            return self._path_completions(text, state, ('.png','.jpg','.jpeg'))

        if buffer.strip().endswith('cd') or text.startswith(('.', '/')):
            return self._path_completions(text, state, ())

        response = None
        if state == 0:
            # This is the first time for this text, so build a match list.
            if text:
                self.matches = [
                    s for s in self.options if s and s.startswith(text)
                ]
            else:
                self.matches = self.options[:]

        # Return the state'th item from the match list,
        # if we have that many.
        try:
            response = self.matches[state]
        except IndexError:
            response = None
        return response

    def _path_completions(self, text, state, extensions):
        # get the path so far
        # TODO: replace this mess with a regular expression match
        if text.startswith('-I'):
            path = text.replace('-I', '', 1).lstrip()
        elif text.startswith('--init_img='):
            path = text.replace('--init_img=', '', 1).lstrip()
        elif text.startswith('--init_mask='):
            path = text.replace('--init_mask=', '', 1).lstrip()
        elif text.startswith('-M'):
            path = text.replace('-M', '', 1).lstrip()
        else:
            path = text

        matches = list()

        path = os.path.expanduser(path)
        if len(path) == 0:
            matches.append(text + './')
        else:
            dir = os.path.dirname(path)
            dir_list = os.listdir(dir)
            for n in dir_list:
                if n.startswith('.') and len(n) > 1:
                    continue
                full_path = os.path.join(dir, n)
                if full_path.startswith(path):
                    if os.path.isdir(full_path):
                        matches.append(
                            os.path.join(os.path.dirname(text), n) + '/'
                        )
                    elif n.endswith(extensions):
                        matches.append(os.path.join(os.path.dirname(text), n))

        try:
            response = matches[state]
        except IndexError:
            response = None
        return response


if readline_available:
    readline.set_completer(
        Completer(
            [
                '--steps','-s',
                '--seed','-S',
                '--iterations','-n',
                '--width','-W','--height','-H',
                '--cfg_scale','-C',
                '--grid','-g',
                '--individual','-i',
                '--init_img','-I',
                '--init_mask','-M',
                '--strength','-f',
                '--variants','-v',
                '--outdir','-o',
                '--sampler','-A','-m',
                '--embedding_path',
                '--device',
                '--grid','-g',
                '--gfpgan_strength','-G',
                '--upscale','-U',
                '-save_orig','--save_original',
                '--skip_normalize','-x',
                '--log_tokenization','t',
            ]
        ).complete
    )
    readline.set_completer_delims(' ')
    readline.parse_and_bind('tab: complete')

    histfile = os.path.join(os.path.expanduser('~'), '.dream_history')
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)
