import { Middleware } from '@reduxjs/toolkit';
import { io } from 'socket.io-client';

import makeSocketIOListeners from './listeners';
import makeSocketIOEmitters from './emitters';

import * as InvokeAI from '../invokeai';

/**
 * Creates a socketio middleware to handle communication with server.
 *
 * Special `socketio/actionName` actions are created in actions.ts and
 * exported for use by the application, which treats them like any old
 * action, using `dispatch` to dispatch them.
 *
 * These actions are intercepted here, where `socketio.emit()` calls are
 * made on their behalf - see `emitters.ts`. The emitter functions
 * are the outbound communication to the server.
 *
 * Listeners are also established here - see `listeners.ts`. The listener
 * functions receive communication from the server and usually dispatch
 * some new action to handle whatever data was sent from the server.
 */
export const socketioMiddleware = () => {
  const { hostname, port } = new URL(window.location.href);

  const socketio = io(`http://${hostname}:${port}`, {
    timeout: 60000,
  });

  let areListenersSet = false;

  const middleware: Middleware = (store) => (next) => (action) => {
    const {
      onConnect,
      onDisconnect,
      onError,
      onPostprocessingResult,
      onGenerationResult,
      onIntermediateResult,
      onProgressUpdate,
      onGalleryImages,
      onProcessingCanceled,
      onImageDeleted,
      onInitialImageUploaded,
      onMaskImageUploaded,
      onSystemConfig,
    } = makeSocketIOListeners(store);

    const {
      emitGenerateImage,
      emitRunESRGAN,
      emitRunGFPGAN,
      emitDeleteImage,
      emitRequestImages,
      emitRequestNewImages,
      emitCancelProcessing,
      emitUploadInitialImage,
      emitUploadMaskImage,
      emitRequestSystemConfig,
    } = makeSocketIOEmitters(store, socketio);

    /**
     * If this is the first time the middleware has been called (e.g. during store setup),
     * initialize all our socket.io listeners.
     */
    if (!areListenersSet) {
      socketio.on('connect', () => onConnect());

      socketio.on('disconnect', () => onDisconnect());

      socketio.on('error', (data: InvokeAI.ErrorResponse) => onError(data));

      socketio.on('generationResult', (data: InvokeAI.ImageResultResponse) =>
        onGenerationResult(data)
      );

      socketio.on(
        'postprocessingResult',
        (data: InvokeAI.ImageResultResponse) => onPostprocessingResult(data)
      );

      socketio.on('intermediateResult', (data: InvokeAI.ImageResultResponse) =>
        onIntermediateResult(data)
      );

      socketio.on('progressUpdate', (data: InvokeAI.SystemStatus) =>
        onProgressUpdate(data)
      );

      socketio.on('galleryImages', (data: InvokeAI.GalleryImagesResponse) =>
        onGalleryImages(data)
      );

      socketio.on('processingCanceled', () => {
        onProcessingCanceled();
      });

      socketio.on('imageDeleted', (data: InvokeAI.ImageUrlAndUuidResponse) => {
        onImageDeleted(data);
      });

      socketio.on('initialImageUploaded', (data: InvokeAI.ImageUrlResponse) => {
        onInitialImageUploaded(data);
      });

      socketio.on('maskImageUploaded', (data: InvokeAI.ImageUrlResponse) => {
        onMaskImageUploaded(data);
      });

      socketio.on('systemConfig', (data: InvokeAI.SystemConfig) => {
        onSystemConfig(data);
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

      case 'socketio/requestImages': {
        emitRequestImages();
        break;
      }

      case 'socketio/requestNewImages': {
        emitRequestNewImages();
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

      case 'socketio/requestSystemConfig': {
        emitRequestSystemConfig();
        break;
      }
    }

    next(action);
  };

  return middleware;
};
