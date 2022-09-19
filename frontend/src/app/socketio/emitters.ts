import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';
import { Socket } from 'socket.io-client';
import { frontendToBackendParameters } from '../../common/util/parameterTranslation';
import { SDImage } from '../../features/gallery/gallerySlice';
import {
  addLogEntry,
  setIsProcessing,
} from '../../features/system/systemSlice';

/**
 * Returns an object containing all functions which use `socketio.emit()`.
 * i.e. those which make server requests.
 */
const makeSocketIOEmitters = (
  store: MiddlewareAPI<Dispatch<AnyAction>, any>,
  socketio: Socket
) => {
  // We need to dispatch actions to redux and get pieces of state from the store.
  const { dispatch, getState } = store;

  return {
    emitGenerateImage: () => {
      dispatch(setIsProcessing(true));

      const { generationParameters, esrganParameters, gfpganParameters } =
        frontendToBackendParameters(getState().sd, getState().system);

      socketio.emit(
        'generateImage',
        generationParameters,
        esrganParameters,
        gfpganParameters
      );

      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `Image generation requested: ${JSON.stringify({
            ...generationParameters,
            ...esrganParameters,
            ...gfpganParameters,
          })}`,
        })
      );
    },
    emitRunESRGAN: (imageToProcess: SDImage) => {
      dispatch(setIsProcessing(true));
      const { upscalingLevel, upscalingStrength } = getState().sd;
      const esrganParameters = {
        upscale: [upscalingLevel, upscalingStrength],
      };
      socketio.emit('runESRGAN', imageToProcess, esrganParameters);
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `ESRGAN upscale requested: ${JSON.stringify({
            file: imageToProcess.url,
            ...esrganParameters,
          })}`,
        })
      );
    },
    emitRunGFPGAN: (imageToProcess: SDImage) => {
      dispatch(setIsProcessing(true));
      const { gfpganStrength } = getState().sd;

      const gfpganParameters = {
        gfpgan_strength: gfpganStrength,
      };
      socketio.emit('runGFPGAN', imageToProcess, gfpganParameters);
      dispatch(
        addLogEntry({
          timestamp: dateFormat(new Date(), 'isoDateTime'),
          message: `GFPGAN fix faces requested: ${JSON.stringify({
            file: imageToProcess.url,
            ...gfpganParameters,
          })}`,
        })
      );
    },
    emitDeleteImage: (imageToDelete: SDImage) => {
      const { url, uuid } = imageToDelete;
      socketio.emit('deleteImage', url, uuid);
    },
    emitRequestAllImages: () => {
      socketio.emit('requestAllImages');
    },
    emitCancelProcessing: () => {
      socketio.emit('cancel');
    },
    emitUploadInitialImage: (file: File) => {
      socketio.emit('uploadInitialImage', file, file.name);
    },
    emitUploadMaskImage: (file: File) => {
      socketio.emit('uploadMaskImage', file, file.name);
    },
  };
};

export default makeSocketIOEmitters;
