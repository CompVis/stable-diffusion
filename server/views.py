# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

"""Views module."""
import json
import os
from queue import Queue
from flask import current_app, jsonify, request, Response, send_from_directory, stream_with_context, url_for
from flask.views import MethodView
from dependency_injector.wiring import inject, Provide

from server.models import DreamResult, JobRequest
from server.services import GeneratorService, ImageStorageService, JobQueueService
from server.containers import Container

class ApiJobs(MethodView):

  @inject
  def post(self, job_queue_service: JobQueueService = Provide[Container.generation_queue_service]):
    jobRequest = JobRequest.from_json(request.json)

    print(f">> Request to generate with prompt: {jobRequest.prompt}")

    # Push the request
    job_queue_service.push(jobRequest)

    return { 'jobId': jobRequest.id }
  

class WebIndex(MethodView):
  init_every_request = False
  __file: str = None
  
  def __init__(self, file):
    self.__file = file

  def get(self):
    return current_app.send_static_file(self.__file)


class WebConfig(MethodView):
  init_every_request = False

  def get(self):
    # unfortunately this import can't be at the top level, since that would cause a circular import
    from ldm.gfpgan.gfpgan_tools import gfpgan_model_exists
    config = {
        'gfpgan_model_exists': gfpgan_model_exists
    }
    js = f"let config = {json.dumps(config)};\n"
    return Response(js, mimetype="application/javascript")


class ApiCancel(MethodView):
  init_every_request = False
  
  @inject
  def get(self, generator_service: GeneratorService = Provide[Container.generator_service]):
    generator_service.cancel()
    return Response(status=204)


# TODO: Combine all image storage access
class ApiImages(MethodView):
  init_every_request = False
  __pathRoot = None
  __storage: ImageStorageService

  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId):
    name = self.__storage.path(dreamId)
    fullpath=os.path.join(self.__pathRoot, name)
    return send_from_directory(os.path.dirname(fullpath), os.path.basename(fullpath))
  
  def delete(self, dreamId):
    result = self.__storage.delete(dreamId)
    return Response(status=204) if result else Response(status=404)


class ApiImagesMetadata(MethodView):
  init_every_request = False
  __pathRoot = None
  __storage: ImageStorageService

  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId):
    meta = self.__storage.getMetadata(dreamId)
    j = {} if meta is None else meta.__dict__
    return j


class ApiIntermediates(MethodView):
  init_every_request = False
  __pathRoot = None
  __storage: ImageStorageService = Provide[Container.image_intermediates_storage_service]

  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_intermediates_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId, step):
    name = self.__storage.path(dreamId, postfix=f'.{step}')
    fullpath=os.path.join(self.__pathRoot, name)
    return send_from_directory(os.path.dirname(fullpath), os.path.basename(fullpath))

  def delete(self, dreamId):
    result = self.__storage.delete(dreamId)
    return Response(status=204) if result else Response(status=404)

    
class ApiImagesList(MethodView):
  init_every_request = False
  __storage: ImageStorageService

  @inject
  def __init__(self, storage: ImageStorageService = Provide[Container.image_storage_service]):
    self.__storage = storage

  def get(self):
    page = request.args.get("page", default=0, type=int)
    perPage = request.args.get("per_page", default=10, type=int)

    result = self.__storage.list_files(page, perPage)
    return result.__dict__
