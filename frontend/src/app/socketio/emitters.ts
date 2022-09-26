import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';
import { Socket } from 'socket.io-client';
import { frontendToBackendParameters } from '../../common/util/parameterTranslation';
import {
  addLogEntry,
  setIsProcessing,
} from '../../features/system/systemSlice';
import * as InvokeAI from '../invokeai';

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
        frontendToBackendParameters(getState().options, getState().system);

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
    emitRunESRGAN: (imageToProcess: InvokeAI.Image) => {
      dispatch(setIsProcessing(true));
      const { upscalingLevel, upscalingStrength } = getState().options;
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
    emitRunGFPGAN: (imageToProcess: InvokeAI.Image) => {
      dispatch(setIsProcessing(true));
      const { gfpganStrength } = getState().options;

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
    emitDeleteImage: (imageToDelete: InvokeAI.Image) => {
      const { url, uuid } = imageToDelete;
      socketio.emit('deleteImage', url, uuid);
    },
    emitRequestImages: () => {
      const { nextPage, offset } = getState().gallery;
      socketio.emit('requestImages', nextPage, offset);
    },
    emitRequestNewImages: () => {
      const { nextPage, offset, images } = getState().gallery;
      if (images.length > 0) {
        socketio.emit('requestImages', nextPage, offset, images[0].mtime);
      } else {
        socketio.emit('requestImages', nextPage, offset);
      }
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
    emitRequestSystemConfig: () => {
      socketio.emit('requestSystemConfig');
    },
  };
};

export default makeSocketIOEmitters;
