# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from base64 import urlsafe_b64encode
import json
import string
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Union
from uuid import uuid4


class DreamBase():
  # Id
  id: str

  # Initial Image
  enable_init_image: bool
  initimg: string = None

  # Img2Img
  enable_img2img: bool # TODO: support this better
  strength: float = 0 # TODO: name this something related to img2img to make it clearer?
  fit = None # Fit initial image dimensions

  # Generation
  enable_generate: bool
  prompt: string = ""
  seed: int = 0 # 0 is random
  steps: int = 10
  width: int = 512
  height: int = 512
  cfg_scale: float = 7.5
  threshold: float = 0.0
  perlin: float = 0.0
  sampler_name: string = 'klms'
  seamless: bool = False
  hires_fix: bool = False
  model: str = None # The model to use (currently unused)
  embeddings = None # The embeddings to use (currently unused)
  progress_images: bool = False

  # GFPGAN
  enable_gfpgan: bool
  facetool_strength: float = 0

  # Upscale
  enable_upscale: bool
  upscale: None
  upscale_level: int = None
  upscale_strength: float = 0.75

  # Embiggen
  enable_embiggen: bool
  embiggen: Union[None, List[float]] = None
  embiggen_tiles: Union[None, List[int]] = None

  # Metadata
  time: int

  def __init__(self):
    self.id = urlsafe_b64encode(uuid4().bytes).decode('ascii')

  def parse_json(self, j, new_instance=False):
    # Id
    if 'id' in j and not new_instance:
      self.id = j.get('id')

    # Initial Image
    self.enable_init_image = 'enable_init_image' in j and bool(j.get('enable_init_image'))
    if self.enable_init_image:
      self.initimg = j.get('initimg')

      # Img2Img
      self.enable_img2img = 'enable_img2img' in j and bool(j.get('enable_img2img'))
      if self.enable_img2img:
        self.strength = float(j.get('strength'))
        self.fit    = 'fit' in j

    # Generation
    self.enable_generate = 'enable_generate' in j and bool(j.get('enable_generate'))
    if self.enable_generate:
      self.prompt = j.get('prompt')
      self.seed = int(j.get('seed'))
      self.steps = int(j.get('steps'))
      self.width = int(j.get('width'))
      self.height = int(j.get('height'))
      self.cfg_scale = float(j.get('cfgscale') or j.get('cfg_scale'))
      self.threshold = float(j.get('threshold'))
      self.perlin = float(j.get('perlin'))
      self.sampler_name  = j.get('sampler') or j.get('sampler_name')
      # model: str = None # The model to use (currently unused)
      # embeddings = None # The embeddings to use (currently unused)
      self.seamless = 'seamless' in j
      self.hires_fix = 'hires_fix' in j
      self.progress_images = 'progress_images' in j

    # GFPGAN
    self.enable_gfpgan = 'enable_gfpgan' in j and bool(j.get('enable_gfpgan'))
    if self.enable_gfpgan:
      self.facetool_strength = float(j.get('facetool_strength'))

    # Upscale
    self.enable_upscale = 'enable_upscale' in j and bool(j.get('enable_upscale'))
    if self.enable_upscale:
      self.upscale_level    = j.get('upscale_level')
      self.upscale_strength = j.get('upscale_strength')
      self.upscale = None if self.upscale_level in {None,''} else [int(self.upscale_level),float(self.upscale_strength)]

    # Embiggen
    self.enable_embiggen = 'enable_embiggen' in j and bool(j.get('enable_embiggen'))
    if self.enable_embiggen:
      self.embiggen       = j.get('embiggen')
      self.embiggen_tiles = j.get('embiggen_tiles')

    # Metadata
    self.time = int(j.get('time')) if ('time' in j and not new_instance) else int(datetime.now(timezone.utc).timestamp())


class DreamResult(DreamBase):
  # Result
  has_upscaled: False
  has_gfpgan: False

  # TODO: use something else for state tracking
  images_generated: int = 0
  images_upscaled: int = 0

  def __init__(self):
    super().__init__()

  def clone_without_img(self):
    copy = deepcopy(self)
    copy.initimg = None
    return copy

  def to_json(self):
    copy = deepcopy(self)
    copy.initimg = None
    j = json.dumps(copy.__dict__)
    return j

  @staticmethod
  def from_json(j, newTime: bool = False):
    d = DreamResult()
    d.parse_json(j)
    return d


# TODO: switch this to a pipelined request, with pluggable steps
# Will likely require generator code changes to accomplish
class JobRequest(DreamBase):
  # Iteration
  iterations: int = 1
  variation_amount = None
  with_variations = None

  # Results
  results: List[DreamResult] = []

  def __init__(self):
    super().__init__()

  def newDreamResult(self) -> DreamResult:
    result = DreamResult()
    result.parse_json(self.__dict__, new_instance=True)
    return result

  @staticmethod
  def from_json(j):
    job = JobRequest()
    job.parse_json(j)

    # Metadata
    job.time = int(j.get('time')) if ('time' in j) else int(datetime.now(timezone.utc).timestamp())

    # Iteration
    if job.enable_generate:
      job.iterations = int(j.get('iterations'))
      job.variation_amount = float(j.get('variation_amount'))
      job.with_variations = j.get('with_variations')

    return job


class ProgressType(Enum):
  GENERATION = 1
  UPSCALING_STARTED = 2
  UPSCALING_DONE = 3

class Signal():
  event: str
  data = None
  room: str = None
  broadcast: bool = False

  def __init__(self, event: str, data, room: str = None, broadcast: bool = False):
    self.event = event
    self.data = data
    self.room = room
    self.broadcast = broadcast

  @staticmethod
  def image_progress(jobId: str, dreamId: str, step: int, totalSteps: int, progressType: ProgressType = ProgressType.GENERATION, hasProgressImage: bool = False):
    return Signal('dream_progress', {
      'jobId': jobId,
      'dreamId': dreamId,
      'step': step,
      'totalSteps': totalSteps,
      'hasProgressImage': hasProgressImage,
      'progressType': progressType.name
    }, room=jobId, broadcast=True)

  # TODO: use a result id or something? Like a sub-job
  @staticmethod
  def image_result(jobId: str, dreamId: str, dreamResult: DreamResult):
    return Signal('dream_result', {
      'jobId': jobId,
      'dreamId': dreamId,
      'dreamRequest': dreamResult.clone_without_img().__dict__
    }, room=jobId, broadcast=True)

  @staticmethod
  def job_started(jobId: str):
    return Signal('job_started', {
      'jobId': jobId
    }, room=jobId, broadcast=True)
    
  @staticmethod
  def job_done(jobId: str):
    return Signal('job_done', {
      'jobId': jobId
    }, room=jobId, broadcast=True)

  @staticmethod
  def job_canceled(jobId: str):
    return Signal('job_canceled', {
      'jobId': jobId
    }, room=jobId, broadcast=True)


class PaginatedItems():
  items: List[Any]
  page: int # Current Page
  pages: int # Total number of pages
  per_page: int # Number of items per page
  total: int # Total number of items in result

  def __init__(self, items: List[Any], page: int, pages: int, per_page: int, total: int):
    self.items = items
    self.page = page
    self.pages = pages
    self.per_page = per_page
    self.total = total

  def to_json(self):
    return json.dumps(self.__dict__)
