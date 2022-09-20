import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { clamp } from 'lodash';
import * as InvokeAI from '../../app/invokeai';

export interface GalleryState {
  currentImage?: InvokeAI.Image;
  currentImageUuid: string;
  images: Array<InvokeAI.Image>;
  intermediateImage?: InvokeAI.Image;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: [],
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.currentImage = action.payload;
      state.currentImageUuid = action.payload.uuid;
    },
    removeImage: (state, action: PayloadAction<string>) => {
      const uuid = action.payload;

      const newImages = state.images.filter((image) => image.uuid !== uuid);

      if (uuid === state.currentImageUuid) {
        /**
         * We are deleting the currently selected image.
         *
         * We want the new currentl selected image to be under the cursor in the
         * gallery, so we need to do some fanagling. The currently selected image
         * is set by its UUID, not its index in the image list.
         *
         * Get the currently selected image's index.
         */
        const imageToDeleteIndex = state.images.findIndex(
          (image) => image.uuid === uuid
        );

        /**
         * New current image needs to be in the same spot, but because the gallery
         * is sorted in reverse order, the new current image's index will actuall be
         * one less than the deleted image's index.
         *
         * Clamp the new index to ensure it is valid..
         */
        const newCurrentImageIndex = clamp(
          imageToDeleteIndex - 1,
          0,
          newImages.length - 1
        );

        state.currentImage = newImages.length
          ? newImages[newCurrentImageIndex]
          : undefined;

        state.currentImageUuid = newImages.length
          ? newImages[newCurrentImageIndex].uuid
          : '';
      }

      state.images = newImages;
    },
    addImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.images.push(action.payload);
      state.currentImageUuid = action.payload.uuid;
      state.intermediateImage = undefined;
      state.currentImage = action.payload;
    },
    setIntermediateImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    setGalleryImages: (state, action: PayloadAction<Array<InvokeAI.Image>>) => {
      const newImages = action.payload;
      if (newImages.length) {
        const newCurrentImage = newImages[newImages.length - 1];
        state.images = newImages;
        state.currentImage = newCurrentImage;
        state.currentImageUuid = newCurrentImage.uuid;
      }
    },
  },
});

export const {
  addImage,
  clearIntermediateImage,
  removeImage,
  setCurrentImage,
  setGalleryImages,
  setIntermediateImage,
} = gallerySlice.actions;

export default gallerySlice.reducer;
