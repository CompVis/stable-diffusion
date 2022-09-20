import { createAction } from '@reduxjs/toolkit';
import * as InvokeAI from '../invokeai';

/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

export const generateImage = createAction<undefined>('socketio/generateImage');
export const runESRGAN = createAction<InvokeAI.Image>('socketio/runESRGAN');
export const runGFPGAN = createAction<InvokeAI.Image>('socketio/runGFPGAN');
export const deleteImage = createAction<InvokeAI.Image>('socketio/deleteImage');
export const requestAllImages = createAction<undefined>(
  'socketio/requestAllImages'
);
export const cancelProcessing = createAction<undefined>(
  'socketio/cancelProcessing'
);
export const uploadInitialImage = createAction<File>(
  'socketio/uploadInitialImage'
);
export const uploadMaskImage = createAction<File>('socketio/uploadMaskImage');

export const requestSystemConfig = createAction<undefined>('socketio/requestSystemConfig');
