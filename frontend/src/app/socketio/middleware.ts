import { Middleware } from '@reduxjs/toolkit';
import { io } from 'socket.io-client';

import makeSocketIOListeners from './listeners';
import makeSocketIOEmitters from './emitters';

import type {
  ServerGenerationResult,
  ServerESRGANResult,
  ServerGFPGANResult,
  ServerIntermediateResult,
  ServerError,
  ServerGalleryImages,
  ServerImageUrlAndUuid,
  ServerImageUrl,
} from './types';
import { SystemStatus } from '../../features/system/systemSlice';

export const socketioMiddleware = () => {
  const { hostname, port } = new URL(window.location.href);

  const socketio = io(`http://${hostname}:9090`);

  let areListenersSet = false;

  const middleware: Middleware = (store) => (next) => (action) => {
    const {
      onConnect,
      onDisconnect,
      onError,
      onESRGANResult,
      onGFPGANResult,
      onGenerationResult,
      onIntermediateResult,
      onProgressUpdate,
      onGalleryImages,
      onProcessingCanceled,
      onImageDeleted,
      onInitialImageUploaded,
      onMaskImageUploaded,
    } = makeSocketIOListeners(store);

    const {
      emitGenerateImage,
      emitRunESRGAN,
      emitRunGFPGAN,
      emitDeleteImage,
      emitRequestAllImages,
      emitCancelProcessing,
      emitUploadInitialImage,
      emitUploadMaskImage,
    } = makeSocketIOEmitters(store, socketio);

    /**
     * If this is the first time the middleware has been called (e.g. during store setup),
     * initialize all our socket.io listeners.
     */
    if (!areListenersSet) {
      socketio.on('connect', () => onConnect());

      socketio.on('disconnect', () => onDisconnect());

      socketio.on('error', (data: ServerError) => onError(data));

      socketio.on('generationResult', (data: ServerGenerationResult) =>
        onGenerationResult(data)
      );

      socketio.on('esrganResult', (data: ServerESRGANResult) =>
        onESRGANResult(data)
      );

      socketio.on('gfpganResult', (data: ServerGFPGANResult) =>
        onGFPGANResult(data)
      );

      socketio.on('intermediateResult', (data: ServerIntermediateResult) =>
        onIntermediateResult(data)
      );

      socketio.on('progressUpdate', (data: SystemStatus) =>
        onProgressUpdate(data)
      );

      socketio.on('galleryImages', (data: ServerGalleryImages) =>
        onGalleryImages(data)
      );

      socketio.on('processingCanceled', () => {
        onProcessingCanceled();
      });

      socketio.on('imageDeleted', (data: ServerImageUrlAndUuid) => {
        onImageDeleted(data);
      });

      socketio.on('initialImageUploaded', (data: ServerImageUrl) => {
        onInitialImageUploaded(data);
      });

      socketio.on('maskImageUploaded', (data: ServerImageUrl) => {
        onMaskImageUploaded(data);
      });

      areListenersSet = true;
    }

    /**
     * Handle redux actions caught by middleware.
     */
    switch (action.type) {
      case 'socketio/generateImage': {
        emitGenerateImage();
        break;
      }

      case 'socketio/runESRGAN': {
        emitRunESRGAN(action.payload);
        break;
      }

      case 'socketio/runGFPGAN': {
        emitRunGFPGAN(action.payload);
        break;
      }

      case 'socketio/deleteImage': {
        emitDeleteImage(action.payload);
        break;
      }

      case 'socketio/requestAllImages': {
        emitRequestAllImages();
        break;
      }

      case 'socketio/cancelProcessing': {
        emitCancelProcessing();
        break;
      }

      case 'socketio/uploadInitialImage': {
        emitUploadInitialImage(action.payload);
        break;
      }

      case 'socketio/uploadMaskImage': {
        emitUploadMaskImage(action.payload);
        break;
      }
    }

    next(action);
  };

  return middleware;
};
