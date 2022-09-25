# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from argparse import ArgumentParser
import base64
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
from threading import Thread
import time
from flask_socketio import SocketIO, join_room, leave_room
from ldm.dream.args import Args
from ldm.dream.generator import embiggen
from PIL import Image

from ldm.dream.pngwriter import PngWriter
from ldm.dream.server import CanceledException
from ldm.generate import Generate
from server.models import DreamResult, JobRequest, PaginatedItems, ProgressType, Signal

class JobQueueService:
  __queue: Queue = Queue()

  def push(self, dreamRequest: DreamResult):
    self.__queue.put(dreamRequest)

  def get(self, timeout: float = None) -> DreamResult:
    return self.__queue.get(timeout= timeout)

class SignalQueueService:
  __queue: Queue = Queue()

  def push(self, signal: Signal):
    self.__queue.put(signal)

  def get(self) -> Signal:
    return self.__queue.get(block=False)


class SignalService:
  __socketio: SocketIO
  __queue: SignalQueueService

  def __init__(self, socketio: SocketIO, queue: SignalQueueService):
    self.__socketio = socketio
    self.__queue = queue

    def on_join(data):
      room = data['room']
      join_room(room)
      self.__socketio.emit("test", "something", room=room)
      
    def on_leave(data):
      room = data['room']
      leave_room(room)

    self.__socketio.on_event('join_room', on_join)
    self.__socketio.on_event('leave_room', on_leave)

    self.__socketio.start_background_task(self.__process)

  def __process(self):
    # preload the model
    print('Started signal queue processor')
    try:
      while True:
        try:
          signal = self.__queue.get()
          self.__socketio.emit(signal.event, signal.data, room=signal.room, broadcast=signal.broadcast)
        except Empty:
          pass
        finally:
          self.__socketio.sleep(0.001)

    except KeyboardInterrupt:
        print('Signal queue processor stopped')


  def emit(self, signal: Signal):
    self.__queue.push(signal)


# TODO: Name this better?
# TODO: Logging and signals should probably be event based (multiple listeners for an event)
class LogService:
  __location: str
  __logFile: str

  def __init__(self, location:str, file:str):
    self.__location = location
    self.__logFile = file

  def log(self, dreamResult: DreamResult, seed = None, upscaled = False):
    with open(os.path.join(self.__location, self.__logFile), "a") as log:
      log.write(f"{dreamResult.id}: {dreamResult.to_json()}\n")


class ImageStorageService:
  __location: str
  __pngWriter: PngWriter
  __legacyParser: ArgumentParser

  def __init__(self, location):
    self.__location = location
    self.__pngWriter = PngWriter(self.__location)
    self.__legacyParser = Args() # TODO: inject this?

  def __getName(self, dreamId: str, postfix: str = '') -> str:
    return f'{dreamId}{postfix}.png'

  def save(self, image, dreamResult: DreamResult, postfix: str = '') -> str:
    name = self.__getName(dreamResult.id, postfix)
    meta = dreamResult.to_json() # TODO: make all methods consistent with writing metadata. Standardize metadata.
    path = self.__pngWriter.save_image_and_prompt_to_png(image, dream_prompt=meta, metadata=None, name=name)
    return path

  def path(self, dreamId: str, postfix: str = '') -> str:
    name = self.__getName(dreamId, postfix)
    path = os.path.join(self.__location, name)
    return path
  
  # Returns true if found, false if not found or error
  def delete(self, dreamId: str, postfix: str = '') -> bool:
    path = self.path(dreamId, postfix)
    if (os.path.exists(path)):
      os.remove(path)
      return True
    else:
      return False
  
  def getMetadata(self, dreamId: str, postfix: str = '') -> DreamResult:
    path = self.path(dreamId, postfix)
    image = Image.open(path)
    text = image.text
    if text.__contains__('Dream'):
      dreamMeta = text.get('Dream')
      try:
        j = json.loads(dreamMeta)
        return DreamResult.from_json(j)
      except ValueError:
        # Try to parse command-line format (legacy metadata format)
        try:
          opt = self.__parseLegacyMetadata(dreamMeta)
          optd = opt.__dict__
          if (not 'width' in optd) or (optd.get('width') is None):
            optd['width'] = image.width
          if (not 'height' in optd) or (optd.get('height') is None):
            optd['height'] = image.height
          if (not 'steps' in optd) or (optd.get('steps') is None):
            optd['steps'] = 10 # No way around this unfortunately - seems like it wasn't storing this previously

          optd['time'] = os.path.getmtime(path) # Set timestamp manually (won't be exactly correct though)

          return DreamResult.from_json(optd)

        except:
          return None
    else:
      return None

  def __parseLegacyMetadata(self, command: str) -> DreamResult:
    # before splitting, escape single quotes so as not to mess
    # up the parser
    command = command.replace("'", "\\'")

    try:
        elements = shlex.split(command)
    except ValueError as e:
        return None

    # rearrange the arguments to mimic how it works in the Dream bot.
    switches = ['']
    switches_started = False

    for el in elements:
        if el[0] == '-' and not switches_started:
            switches_started = True
        if switches_started:
            switches.append(el)
        else:
            switches[0] += el
            switches[0] += ' '
    switches[0] = switches[0][: len(switches[0]) - 1]

    try:
        opt = self.__legacyParser.parse_cmd(switches)
        return opt
    except SystemExit:
        return None

  def list_files(self, page: int, perPage: int) -> PaginatedItems:
    files = sorted(glob.glob(os.path.join(self.__location,'*.png')), key=os.path.getmtime, reverse=True)
    count = len(files)

    startId = page * perPage
    pageCount = int(count / perPage) + 1
    endId = min(startId + perPage, count)
    items = [] if startId >= count else files[startId:endId]

    items = list(map(lambda f: Path(f).stem, items))

    return PaginatedItems(items, page, pageCount, perPage, count)


class GeneratorService:
  __model: Generate
  __queue: JobQueueService
  __imageStorage: ImageStorageService
  __intermediateStorage: ImageStorageService
  __log: LogService
  __thread: Thread
  __cancellationRequested: bool = False
  __signal_service: SignalService

  def __init__(self, model: Generate, queue: JobQueueService, imageStorage: ImageStorageService, intermediateStorage: ImageStorageService, log: LogService, signal_service: SignalService):
    self.__model = model
    self.__queue = queue
    self.__imageStorage = imageStorage
    self.__intermediateStorage = intermediateStorage
    self.__log = log
    self.__signal_service = signal_service

    # Create the background thread
    self.__thread = Thread(target=self.__process, name = "GeneratorService")
    self.__thread.daemon = True
    self.__thread.start()


  # Request cancellation of the current job
  def cancel(self):
    self.__cancellationRequested = True


  # TODO: Consider moving this to its own service if there's benefit in separating the generator
  def __process(self):
    # preload the model
    # TODO: support multiple models
    print('Preloading model')
    tic = time.time()
    self.__model.load_model()
    print(f'>> model loaded in', '%4.2fs' % (time.time() - tic))

    print('Started generation queue processor')
    try:
      while True:
        dreamRequest = self.__queue.get()
        self.__generate(dreamRequest)

    except KeyboardInterrupt:
        print('Generation queue processor stopped')


  def __on_start(self, jobRequest: JobRequest):
    self.__signal_service.emit(Signal.job_started(jobRequest.id))


  def __on_image_result(self, jobRequest: JobRequest, image, seed, upscaled=False):
    dreamResult = jobRequest.newDreamResult()
    dreamResult.seed = seed
    dreamResult.has_upscaled = upscaled
    dreamResult.iterations = 1
    jobRequest.results.append(dreamResult)
    # TODO: Separate status of GFPGAN?

    self.__imageStorage.save(image, dreamResult)
    
    # TODO: handle upscaling logic better (this is appending data to log, but only on first generation)
    if not upscaled:
      self.__log.log(dreamResult)

    # Send result signal
    self.__signal_service.emit(Signal.image_result(jobRequest.id, dreamResult.id, dreamResult))

    upscaling_requested = dreamResult.enable_upscale or dreamResult.enable_gfpgan
    
    # Report upscaling status
    # TODO: this is very coupled to logic inside the generator. Fix that.
    if upscaling_requested and any(result.has_upscaled for result in jobRequest.results):
      progressType = ProgressType.UPSCALING_STARTED if len(jobRequest.results) < 2 * jobRequest.iterations else ProgressType.UPSCALING_DONE
      upscale_count = sum(1 for i in jobRequest.results if i.has_upscaled)
      self.__signal_service.emit(Signal.image_progress(jobRequest.id, dreamResult.id, upscale_count, jobRequest.iterations, progressType))


  def __on_progress(self, jobRequest: JobRequest, sample, step):
    if self.__cancellationRequested:
      self.__cancellationRequested = False
      raise CanceledException

    # TODO: Progress per request will be easier once the seeds (and ids) can all be pre-generated
    hasProgressImage = False
    s = str(len(jobRequest.results))
    if jobRequest.progress_images and step % 5 == 0 and step < jobRequest.steps - 1:
      image = self.__model._sample_to_image(sample)

      # TODO: clean this up, use a pre-defined dream result
      result = DreamResult()
      result.parse_json(jobRequest.__dict__, new_instance=False)
      self.__intermediateStorage.save(image, result, postfix=f'.{s}.{step}')
      hasProgressImage = True

    self.__signal_service.emit(Signal.image_progress(jobRequest.id, f'{jobRequest.id}.{s}', step, jobRequest.steps, ProgressType.GENERATION, hasProgressImage))


  def __generate(self, jobRequest: JobRequest):
    try:
      # TODO: handle this file a file service for init images
      initimgfile = None # TODO: support this on the model directly?
      if (jobRequest.enable_init_image):
        if jobRequest.initimg is not None:
          with open("./img2img-tmp.png", "wb") as f:
            initimg = jobRequest.initimg.split(",")[1] # Ignore mime type
            f.write(base64.b64decode(initimg))
            initimgfile = "./img2img-tmp.png"

      # Use previous seed if set to -1
      initSeed = jobRequest.seed
      if initSeed == -1:
        initSeed = self.__model.seed

      # Zero gfpgan strength if the model doesn't exist
      # TODO: determine if this could be at the top now? Used to cause circular import
      from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
      if not gfpgan_model_exists:
        jobRequest.enable_gfpgan = False

      # Signal start
      self.__on_start(jobRequest)

      # Generate in model
      # TODO: Split job generation requests instead of fitting all parameters here
      # TODO: Support no generation (just upscaling/gfpgan)

      upscale = None if not jobRequest.enable_upscale else jobRequest.upscale
      gfpgan_strength = 0 if not jobRequest.enable_gfpgan else jobRequest.gfpgan_strength

      if not jobRequest.enable_generate:
        # If not generating, check if we're upscaling or running gfpgan
        if not upscale and not gfpgan_strength:
          # Invalid settings (TODO: Add message to help user)
          raise CanceledException()

        image = Image.open(initimgfile)
        # TODO: support progress for upscale?
        self.__model.upscale_and_reconstruct(
          image_list = [[image,0]],
          upscale = upscale,
          strength = gfpgan_strength,
          save_original = False,
          image_callback = lambda image, seed, upscaled=False: self.__on_image_result(jobRequest, image, seed, upscaled))

      else:
        # Generating - run the generation
        init_img = None if (not jobRequest.enable_img2img or jobRequest.strength == 0) else initimgfile


        self.__model.prompt2image(
          prompt           = jobRequest.prompt,
          init_img         = init_img, # TODO: ensure this works
          strength         = None if init_img is None else jobRequest.strength,
          fit              = None if init_img is None else jobRequest.fit,
          iterations       = jobRequest.iterations,
          cfg_scale        = jobRequest.cfg_scale,
          width            = jobRequest.width,
          height           = jobRequest.height,
          seed             = jobRequest.seed,
          steps            = jobRequest.steps,
          variation_amount = jobRequest.variation_amount,
          with_variations  = jobRequest.with_variations,
          gfpgan_strength  = gfpgan_strength,
          upscale          = upscale,
          sampler_name     = jobRequest.sampler_name,
          seamless         = jobRequest.seamless,
          embiggen         = jobRequest.embiggen,
          embiggen_tiles   = jobRequest.embiggen_tiles,
          step_callback    = lambda sample, step: self.__on_progress(jobRequest, sample, step),
          image_callback   = lambda image, seed, upscaled=False: self.__on_image_result(jobRequest, image, seed, upscaled))

    except CanceledException:
      self.__signal_service.emit(Signal.job_canceled(jobRequest.id))

    finally:
      self.__signal_service.emit(Signal.job_done(jobRequest.id))

      # Remove the temp file
      if (initimgfile is not None):
        os.remove("./img2img-tmp.png")
