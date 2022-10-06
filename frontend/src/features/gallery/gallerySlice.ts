import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import _, { clamp } from 'lodash';
import * as InvokeAI from '../../app/invokeai';

export interface GalleryState {
  currentImage?: InvokeAI.Image;
  currentImageUuid: string;
  images: Array<InvokeAI.Image>;
  intermediateImage?: InvokeAI.Image;
  areMoreImagesAvailable: boolean;
  latest_mtime?: number;
  earliest_mtime?: number;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: [],
  areMoreImagesAvailable: true,
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
      const newImage = action.payload;
      const { uuid, mtime } = newImage;
      state.images.unshift(newImage);
      state.currentImageUuid = uuid;
      state.intermediateImage = undefined;
      state.currentImage = newImage;
      state.latest_mtime = mtime;
    },
    setIntermediateImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    selectNextImage: (state) => {
      const { images, currentImage } = state;
      if (currentImage) {
        const currentImageIndex = images.findIndex(
          (i) => i.uuid === currentImage.uuid
        );
        if (_.inRange(currentImageIndex, 0, images.length)) {
          const newCurrentImage = images[currentImageIndex + 1];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }
      }
    },
    selectPrevImage: (state) => {
      const { images, currentImage } = state;
      if (currentImage) {
        const currentImageIndex = images.findIndex(
          (i) => i.uuid === currentImage.uuid
        );
        if (_.inRange(currentImageIndex, 1, images.length + 1)) {
          const newCurrentImage = images[currentImageIndex - 1];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }
      }
    },
    addGalleryImages: (
      state,
      action: PayloadAction<{
        images: Array<InvokeAI.Image>;
        areMoreImagesAvailable: boolean;
      }>
    ) => {
      const { images, areMoreImagesAvailable } = action.payload;
      if (images.length > 0) {
        state.images = state.images
          .concat(images)
          .sort((a, b) => b.mtime - a.mtime);

        if (!state.currentImage) {
          const newCurrentImage = images[0];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }

        // keep track of the timestamps of latest and earliest images received
        state.latest_mtime = images[0].mtime;
        state.earliest_mtime = images[images.length - 1].mtime;
      }
      if (areMoreImagesAvailable !== undefined) {
        state.areMoreImagesAvailable = areMoreImagesAvailable;
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
  selectNextImage,
  selectPrevImage,
} = gallerySlice.actions;

export default gallerySlice.reducer;
