import { AnyAction, MiddlewareAPI, Dispatch } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import dateFormat from 'dateformat';

import {
  addLogEntry,
  setIsConnected,
  setIsProcessing,
  SystemStatus,
  setSystemStatus,
  setCurrentStatus,
} from '../../features/system/systemSlice';

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

import { backendToFrontendParameters } from '../../common/util/parameterTranslation';

import {
  addImage,
  clearIntermediateImage,
  removeImage,
  SDImage,
  setGalleryImages,
  setIntermediateImage,
} from '../../features/gallery/gallerySlice';

import { setInitialImagePath, setMaskPath } from '../../features/sd/sdSlice';

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
        dispatch(setIsProcessing(false));
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
    onGenerationResult: (data: ServerGenerationResult) => {
      try {
        const { url, metadata } = data;
        const newUuid = uuidv4();

        const translatedMetadata = backendToFrontendParameters(metadata);

        dispatch(
          addImage({
            uuid: newUuid,
            url,
            metadata: translatedMetadata,
          })
        );
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Image generated: ${url}`,
          })
        );
        dispatch(setIsProcessing(false));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'intermediateResult' event.
     */
    onIntermediateResult: (data: ServerIntermediateResult) => {
      try {
        const uuid = uuidv4();
        const { url, metadata } = data;
        dispatch(
          setIntermediateImage({
            uuid,
            url,
            metadata,
          })
        );
        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Intermediate image generated: ${url}`,
          })
        );
        dispatch(setIsProcessing(false));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive an 'esrganResult' event.
     */
    onESRGANResult: (data: ServerESRGANResult) => {
      try {
        const { url, uuid, metadata } = data;
        const newUuid = uuidv4();

        // This image was only ESRGAN'd, grab the original image's metadata
        const originalImage = getState().gallery.images.find(
          (i: SDImage) => i.uuid === uuid
        );

        // Retain the original metadata
        const newMetadata = {
          ...originalImage.metadata,
        };

        // Update the ESRGAN-related fields
        newMetadata.shouldRunESRGAN = true;
        newMetadata.upscalingLevel = metadata.upscale[0];
        newMetadata.upscalingStrength = metadata.upscale[1];

        dispatch(
          addImage({
            uuid: newUuid,
            url,
            metadata: newMetadata,
          })
        );

        dispatch(
          addLogEntry({
            timestamp: dateFormat(new Date(), 'isoDateTime'),
            message: `Upscaled: ${url}`,
          })
        );
        dispatch(setIsProcessing(false));
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'gfpganResult' event.
     */
    onGFPGANResult: (data: ServerGFPGANResult) => {
      try {
        const { url, uuid, metadata } = data;
        const newUuid = uuidv4();

        // This image was only GFPGAN'd, grab the original image's metadata
        const originalImage = getState().gallery.images.find(
          (i: SDImage) => i.uuid === uuid
        );

        // Retain the original metadata
        const newMetadata = {
          ...originalImage.metadata,
        };

        // Update the GFPGAN-related fields
        newMetadata.shouldRunGFPGAN = true;
        newMetadata.gfpganStrength = metadata.gfpgan_strength;

        dispatch(
          addImage({
            uuid: newUuid,
            url,
            metadata: newMetadata,
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
    onProgressUpdate: (data: SystemStatus) => {
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
    onError: (data: ServerError) => {
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
        dispatch(setIsProcessing(false));
        dispatch(clearIntermediateImage());
      } catch (e) {
        console.error(e);
      }
    },
    /**
     * Callback to run when we receive a 'galleryImages' event.
     */
    onGalleryImages: (data: ServerGalleryImages) => {
      const { images } = data;
      const preparedImages = images.map((image): SDImage => {
        return {
          uuid: uuidv4(),
          url: image.path,
          metadata: backendToFrontendParameters(image.metadata),
        };
      });
      dispatch(setGalleryImages(preparedImages));
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
      dispatch(setIsProcessing(false));

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
    onImageDeleted: (data: ServerImageUrlAndUuid) => {
      const { url, uuid } = data;
      dispatch(removeImage(uuid));
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
    onInitialImageUploaded: (data: ServerImageUrl) => {
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
    onMaskImageUploaded: (data: ServerImageUrl) => {
      const { url } = data;
      dispatch(setMaskPath(url));
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Mask image uploaded: ${url}`,
        })
      );
    },
  };
};

export default makeSocketIOListeners;
