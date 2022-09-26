import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { clamp } from 'lodash';
import * as InvokeAI from '../../app/invokeai';

export interface GalleryState {
  currentImage?: InvokeAI.Image;
  currentImageUuid: string;
  images: Array<InvokeAI.Image>;
  intermediateImage?: InvokeAI.Image;
  nextPage: number;
  offset: number;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: [],
  nextPage: 1,
  offset: 0,
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
          imageToDeleteIndex,
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
      state.images.unshift(action.payload);
      state.currentImageUuid = action.payload.uuid;
      state.intermediateImage = undefined;
      state.currentImage = action.payload;
      state.offset += 1
    },
    setIntermediateImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    addGalleryImages: (
      state,
      action: PayloadAction<{
        images: Array<InvokeAI.Image>;
        nextPage: number;
        offset: number;
      }>
    ) => {
      const { images, nextPage, offset } = action.payload;
      if (images.length) {
        const newCurrentImage = images[0];
        state.images = state.images
          .concat(images)
          .sort((a, b) => b.mtime - a.mtime);
        state.currentImage = newCurrentImage;
        state.currentImageUuid = newCurrentImage.uuid;
        state.nextPage = nextPage;
        state.offset = offset;
      }
    },
  },
});

export const {
  addImage,
  clearIntermediateImage,
  removeImage,
  setCurrentImage,
  addGalleryImages,
  setIntermediateImage,
} = gallerySlice.actions;

export default gallerySlice.reducer;
