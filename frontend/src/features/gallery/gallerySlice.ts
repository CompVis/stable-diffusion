import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';
import { UpscalingLevel } from '../sd/sdSlice';
import { backendToFrontendParameters } from '../../app/parameterTranslation';

// TODO: Revise pending metadata RFC: https://github.com/lstein/stable-diffusion/issues/266
export interface SDMetadata {
  prompt?: string;
  steps?: number;
  cfgScale?: number;
  height?: number;
  width?: number;
  sampler?: string;
  seed?: number;
  img2imgStrength?: number;
  gfpganStrength?: number;
  upscalingLevel?: UpscalingLevel;
  upscalingStrength?: number;
  initialImagePath?: string;
  maskPath?: string;
  seamless?: boolean;
  shouldFitToWidthHeight?: boolean;
}

export interface SDImage {
  // TODO: I have installed @types/uuid but cannot figure out how to use them here.
  uuid: string;
  url: string;
  metadata: SDMetadata;
}

export interface GalleryState {
  currentImageUuid: string;
  images: Array<SDImage>;
  intermediateImage?: SDImage;
  currentImage?: SDImage;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: [],
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<SDImage>) => {
      state.currentImage = action.payload;
      state.currentImageUuid = action.payload.uuid;
    },
    removeImage: (state, action: PayloadAction<SDImage>) => {
      const { uuid } = action.payload;

      const newImages = state.images.filter((image) => image.uuid !== uuid);

      const imageToDeleteIndex = state.images.findIndex(
        (image) => image.uuid === uuid
      );

      const newCurrentImageIndex = Math.min(
        Math.max(imageToDeleteIndex, 0),
        newImages.length - 1
      );

      state.images = newImages;

      state.currentImage = newImages.length
        ? newImages[newCurrentImageIndex]
        : undefined;

      state.currentImageUuid = newImages.length
        ? newImages[newCurrentImageIndex].uuid
        : '';
    },
    addImage: (state, action: PayloadAction<SDImage>) => {
      state.images.push(action.payload);
      state.currentImageUuid = action.payload.uuid;
      state.intermediateImage = undefined;
      state.currentImage = action.payload;
    },
    setIntermediateImage: (state, action: PayloadAction<SDImage>) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    setGalleryImages: (
      state,
      action: PayloadAction<
        Array<{
          path: string;
          metadata: { [key: string]: string | number | boolean };
        }>
      >
    ) => {
      // TODO: Revise pending metadata RFC: https://github.com/lstein/stable-diffusion/issues/266
      const images = action.payload;

      if (images.length === 0) {
        // there are no images on disk, clear the gallery
        state.images = [];
        state.currentImageUuid = '';
        state.currentImage = undefined;
      } else {
        // Filter image urls that are already in the rehydrated state
        const filteredImages = action.payload.filter(
          (image) => !state.images.find((i) => i.url === image.path)
        );

        const preparedImages = filteredImages.map((image): SDImage => {
          return {
            uuid: uuidv4(),
            url: image.path,
            metadata: backendToFrontendParameters(image.metadata),
          };
        });

        const newImages = [...state.images].concat(preparedImages);

        // if previous currentimage no longer exists, set a new one
        if (!newImages.find((image) => image.uuid === state.currentImageUuid)) {
          const newCurrentImage = newImages[newImages.length - 1];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }

        state.images = newImages;
      }
    },
  },
});

export const {
  setCurrentImage,
  removeImage,
  addImage,
  setGalleryImages,
  setIntermediateImage,
  clearIntermediateImage,
} = gallerySlice.actions;

export default gallerySlice.reducer;
