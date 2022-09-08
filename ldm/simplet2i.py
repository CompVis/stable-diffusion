'''
This module is provided for backward compatibility with the
original (hasty) API.

Please use ldm.generate instead.
'''

from ldm.generate import Generate

class T2I(Generate):
    def __init__(self,**kwargs):
        print(f'>> The ldm.simplet2i module is deprecated. Use ldm.generate instead. It is a drop-in replacement.')
        super().__init__(kwargs)
