import { createAction } from '@reduxjs/toolkit';
import { SDImage } from '../../features/gallery/gallerySlice';

/**
 * We can't use redux-toolkit's createSlice() to make these actions,
 * because they have no associated reducer. They only exist to dispatch
 * requests to the server via socketio. These actions will be handled
 * by the middleware.
 */

export const generateImage = createAction<undefined>('socketio/generateImage');
export const runESRGAN = createAction<SDImage>('socketio/runESRGAN');
export const runGFPGAN = createAction<SDImage>('socketio/runGFPGAN');
export const deleteImage = createAction<SDImage>('socketio/deleteImage');
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
