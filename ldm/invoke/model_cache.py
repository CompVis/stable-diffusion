'''
Manage a cache of Stable Diffusion model files for fast switching. 
They are moved between GPU and CPU as necessary. If CPU memory falls
below a preset minimum, the least recently used model will be
cleared and loaded from disk when next needed.
'''

import torch
import os
import io
import time
import gc
import hashlib
import psutil
import transformers
from sys import getrefcount
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from ldm.util import instantiate_from_config

GIGS=2**30
AVG_MODEL_SIZE=2.1*GIGS

class ModelCache(object):
    def __init__(self, config:OmegaConf, device_type:str, precision:str, min_free_mem=2*GIGS):
        # prevent nasty-looking CLIP log message
        transformers.logging.set_verbosity_error()
        self.config = config
        self.precision = precision
        self.device = torch.device(device_type)
        self.min_free_mem = min_free_mem
        self.models = {}
        self.stack = []  # this is an LRU FIFO
        self.current_model = None

    def get_model(self, model_name:str):
        if model_name not in self.config:
            print(f'"{model_name}" is not a known model name. Please check your models.yaml file')
            return None

        if self.current_model != model_name:
            self.unload_model(self.current_model)
        
        if model_name in self.models:
            requested_model = self.models[model_name]['model']
            self._model_from_cpu(requested_model)
            width = self.models[model_name]['width']
            height = self.models[model_name]['height']
        else:
            self._check_memory()
            requested_model, width, height = self._load_model(model_name)
            self.models[model_name] = {}
            self.models[model_name]['model'] = requested_model
            self.models[model_name]['width'] = width
            self.models[model_name]['height'] = height

        self.current_model = model_name
        self._push_newest_model(model_name)
        return requested_model, width, height

    def list_models(self):
        for name in self.config:
            try:
                description = self.config[name].description
            except ConfigAttributeError:
                description = '<no description>'
            if self.current_model == name:
                status = 'active'
            elif name in self.models:
                status = 'cached'
            else:
                status = 'not loaded'
            print(f'{name:20s} {status:>10s}  {description}')
            

    def _check_memory(self):
        free_memory = psutil.virtual_memory()[4]
        print(f'DEBUG: free memory = {free_memory}, min_mem = {self.min_free_mem}')
        while free_memory + AVG_MODEL_SIZE < self.min_free_mem:

            print(f'DEBUG: free memory = {free_memory}')
            least_recent_model = self._pop_oldest_model()
            if least_recent_model is None:
                return

            print(f'DEBUG: clearing {least_recent_model} from cache (refcount = {getrefcount(self.models[least_recent_model]["model"])})')
            del self.models[least_recent_model]['model']
            gc.collect()

            new_free_memory = psutil.virtual_memory()[4]
            if new_free_memory <= free_memory:
                print(f'>> **Unable to free memory for model caching.**')
                break;
            free_memory = new_free_memory

        
    def _load_model(self, model_name:str):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if model_name not in self.config:
            print(f'"{model_name}" is not a known model name. Please check your models.yaml file')
            return None

        mconfig = self.config[model_name]
        config = mconfig.config
        weights = mconfig.weights
        width = mconfig.width
        height = mconfig.height

        print(f'>> Loading {model_name} weights from {weights}')

        # for usage statistics
        if self._has_cuda():
            torch.cuda.reset_peak_memory_stats()
        tic = time.time()

        # this does the work
        c     = OmegaConf.load(config)
        with open(weights,'rb') as f:
            weight_bytes = f.read()
        self.model_hash  = self._cached_sha256(weights,weight_bytes)
        pl_sd = torch.load(io.BytesIO(weight_bytes), map_location='cpu')
        del weight_bytes
        sd    = pl_sd['state_dict']
        model = instantiate_from_config(c.model)
        m, u  = model.load_state_dict(sd, strict=False)

        if self.precision == 'float16':
            print('>> Using faster float16 precision')
            model.to(torch.float16)
        else:
            print('>> Using more accurate float32 precision')

        model.to(self.device)
        model.eval()

        # usage statistics
        toc = time.time()
        print(f'>> Model loaded in', '%4.2fs' % (toc - tic))
        if self._has_cuda():
            print(
                '>> Max VRAM used to load the model:',
                '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
                '\n>> Current VRAM usage:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
            )
        return model, width, height
        
    def unload_model(self, model_name:str):
        if model_name not in self.models:
            return
        print(f'>> Unloading model {model_name}')
        model = self.models[model_name]['model']
        self._model_to_cpu(model)
        gc.collect()
        if self._has_cuda():
            torch.cuda.empty_cache()

    def _model_to_cpu(self,model):
        if self._has_cuda():
            print(f'DEBUG: moving model to cpu')
            model.first_stage_model.to('cpu')
            model.cond_stage_model.to('cpu') 
            model.model.to('cpu')

    def _model_from_cpu(self,model):
        if self._has_cuda():
            print(f'DEBUG: moving model into {self.device.type}')
            model.to(self.device)
            model.first_stage_model.to(self.device)
            model.cond_stage_model.to(self.device)

    def _pop_oldest_model(self):
        '''
        Remove the first element of the FIFO, which ought
        to be the least recently accessed model.
        '''
        if len(self.stack)>0:
            self.stack.pop(0)

    def _push_newest_model(self,model_name:str):
        '''
        Maintain a simple FIFO. First element is always the
        least recent, and last element is always the most recent.
        '''
        try:
            self.stack.remove(model_name)
        except ValueError:
            pass
        self.stack.append(model_name)
        print(f'DEBUG, stack={self.stack}')
        
    def _has_cuda(self):
        return self.device.type == 'cuda'

    def _cached_sha256(self,path,data):
        dirname    = os.path.dirname(path)
        basename   = os.path.basename(path)
        base, _    = os.path.splitext(basename)
        hashpath   = os.path.join(dirname,base+'.sha256')
        if os.path.exists(hashpath) and os.path.getmtime(path) <= os.path.getmtime(hashpath):
            with open(hashpath) as f:
                hash = f.read()
            return hash
        print(f'>> Calculating sha256 hash of weights file')
        tic = time.time()
        sha = hashlib.sha256()
        sha.update(data)
        hash = sha.hexdigest()
        toc = time.time()
        print(f'>> sha256 = {hash}','(%4.2fs)' % (toc - tic))
        with open(hashpath,'w') as f:
            f.write(hash)
        return hash
