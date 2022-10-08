import { AnyAction, Dispatch, MiddlewareAPI } from '@reduxjs/toolkit';
import dateFormat from 'dateformat';
import { Socket } from 'socket.io-client';
import { frontendToBackendParameters } from '../../common/util/parameterTranslation';
import {
  addLogEntry,
  setIsProcessing,
} from '../../features/system/systemSlice';
import { tabMap, tab_dict } from '../../features/tabs/InvokeTabs';
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

      const options = { ...getState().options };

      if (tabMap[options.activeTab] === 'txt2img') {
        options.shouldUseInitImage = false;
      }

      const { generationParameters, esrganParameters, gfpganParameters } =
        frontendToBackendParameters(options, getState().system);

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
      socketio.emit('runPostprocessing', imageToProcess, {
        type: 'esrgan',
        ...esrganParameters,
      });
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
      socketio.emit('runPostprocessing', imageToProcess, {
        type: 'gfpgan',
        ...gfpganParameters,
      });
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
      const { earliest_mtime } = getState().gallery;
      socketio.emit('requestImages', earliest_mtime);
    },
    emitRequestNewImages: () => {
      const { latest_mtime } = getState().gallery;
      socketio.emit('requestLatestImages', latest_mtime);
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
