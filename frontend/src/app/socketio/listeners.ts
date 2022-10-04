import { AnyAction, MiddlewareAPI, Dispatch } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import dateFormat from 'dateformat';

import * as InvokeAI from '../invokeai';

import {
  addLogEntry,
  setIsConnected,
  setIsProcessing,
  setSystemStatus,
  setCurrentStatus,
  setSystemConfig,
  processingCanceled,
  errorOccurred,
} from '../../features/system/systemSlice';

import {
  addGalleryImages,
  addImage,
  clearIntermediateImage,
  removeImage,
  setIntermediateImage,
} from '../../features/gallery/gallerySlice';

import {
  setInitialImagePath,
  setMaskPath,
} from '../../features/options/optionsSlice';
import { requestImages, requestNewImages } from './actions';

/**
 * Returns an object containing listener callbacks for socketio events.
 * TODO: This file is large, but simple. Should it be split up further?
 */
const makeSocketIOListeners = (
  store: MiddlewareAPI<Dispatch<AnyAction>, any>
) => {
  const { dispatch, getState } = store;

  return {
    /**
     * Callback to run when we receive a 'connect' event.
     */
    onConnect: () => {
      try {
        dispatch(setIsConnected(true));
        dispatch(setCurrentStatus('Connected'));
        if (getState().gallery.latest_mtime) {
          dispatch(requestNewImages());
        } else {
          dispatch(requestImages());
        }
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'disconnect' event.
     */
    onDisconnect: () => {
      try {
        dispatch(setIsConnected(false));
        dispatch(setCurrentStatus('Disconnected'));

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Disconnected from server`,
            level: 'warning',
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'generationResult' event.
     */
    onGenerationResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        const { url, mtime, metadata } = data;
        const newUuid = uuidv4();

        dispatch(
          addImage({
            uuid: newUuid,
            url,
            mtime,
            metadata: metadata,
          })
        );
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Image generated: ${url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'intermediateResult' event.
     */
    onIntermediateResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        const uuid = uuidv4();
        const { url, metadata, mtime } = data;
        dispatch(
          setIntermediateImage({
            uuid,
            url,
            mtime,
            metadata,
          })
        );
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Intermediate image generated: ${url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive an 'esrganResult' event.
     */
    onPostprocessingResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        const { url, metadata, mtime } = data;

        dispatch(
          addImage({
            uuid: uuidv4(),
            url,
            mtime,
            metadata,
          })
        );

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Postprocessed: ${url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'gfpganResult' event.
     */
    onGFPGANResult: (data: InvokeAI.ImageResultResponse) => {
      try {
        const { url, metadata, mtime } = data;

        dispatch(
          addImage({
            uuid: uuidv4(),
            url,
            mtime,
            metadata,
          })
        );

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Fixed faces: ${url}`,
          })
        );
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     * TODO: Add additional progress phases
     */
    onProgressUpdate: (data: InvokeAI.SystemStatus) => {
      try {
        dispatch(setIsProcessing(true));
        dispatch(setSystemStatus(data));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'progressUpdate' event.
     */
    onError: (data: InvokeAI.ErrorResponse) => {
      const { message, additionalData } = data;

      if (additionalData) {
        // TODO: handle more data than short message
      }

      try {
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Server error: ${message}`,
            level: 'error',
          })
        );
        dispatch(errorOccurred());
        dispatch(clearIntermediateImage());
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'galleryImages' event.
     */
    onGalleryImages: (data: InvokeAI.GalleryImagesResponse) => {
      const { images, areMoreImagesAvailable } = data;

      /**
       * the logic here ideally would be in the reducer but we have a side effect:
       * generating a uuid. so the logic needs to be here, outside redux.
       */

      // Generate a UUID for each image
      const preparedImages = images.map((image): InvokeAI.Image => {
        const { url, metadata, mtime } = image;
        return {
          uuid: uuidv4(),
          url,
          mtime,
          metadata,
        };
      });

      dispatch(
        addGalleryImages({ images: preparedImages, areMoreImagesAvailable })
      );

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Loaded ${images.length} images`,
        })
      );
    },
    /**
     * Callback to run when we receive a 'processingCanceled' event.
     */
    onProcessingCanceled: () => {
      dispatch(processingCanceled());

      const { intermediateImage } = getState().gallery;

      if (intermediateImage) {
        dispatch(addImage(intermediateImage));
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Intermediate image saved: ${intermediateImage.url}`,
          })
        );
        dispatch(clearIntermediateImage());
      }

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Processing canceled`,
          level: 'warning',
        })
      );
    },
    /**
     * Callback to run when we receive a 'imageDeleted' event.
     */
    onImageDeleted: (data: InvokeAI.ImageUrlAndUuidResponse) => {
      const { url, uuid } = data;
      dispatch(removeImage(uuid));

      const { initialImagePath, maskPath } = getState().options;

      if (initialImagePath === url) {
        dispatch(setInitialImagePath(''));
      }

      if (maskPath === url) {
        dispatch(setMaskPath(''));
      }

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Image deleted: ${url}`,
        })
      );
    },
    /**
     * Callback to run when we receive a 'initialImageUploaded' event.
     */
    onInitialImageUploaded: (data: InvokeAI.ImageUrlResponse) => {
      const { url } = data;
      dispatch(setInitialImagePath(url));
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Initial image uploaded: ${url}`,
        })
      );
    },
    /**
     * Callback to run when we receive a 'maskImageUploaded' event.
     */
    onMaskImageUploaded: (data: InvokeAI.ImageUrlResponse) => {
      const { url } = data;
      dispatch(setMaskPath(url));
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Mask image uploaded: ${url}`,
        })
      );
    },
    onSystemConfig: (data: InvokeAI.SystemConfig) => {
      dispatch(setSystemConfig(data));
    },
  };
};

export default makeSocketIOListeners;
