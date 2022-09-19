import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { SDMetadata } from '../gallery/gallerySlice';

export type UpscalingLevel = 2 | 4;

export interface SDState {
  prompt: string;
  iterations: number;
  steps: number;
  cfgScale: number;
  height: number;
  width: number;
  sampler: string;
  threshold: number;
  perlin: number;
  seed: number;
  img2imgStrength: number;
  gfpganStrength: number;
  upscalingLevel: UpscalingLevel;
  upscalingStrength: number;
  shouldUseInitImage: boolean;
  initialImagePath: string;
  maskPath: string;
  seamless: boolean;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  variationAmount: number;
  seedWeights: string;
  shouldRunESRGAN: boolean;
  shouldRunGFPGAN: boolean;
  shouldRandomizeSeed: boolean;
}

const initialSDState: SDState = {
  prompt: '',
  iterations: 1,
  steps: 50,
  cfgScale: 7.5,
  height: 512,
  width: 512,
  sampler: 'k_lms',
  threshold: 0,
  perlin: 0,
  seed: 0,
  seamless: false,
  shouldUseInitImage: false,
  img2imgStrength: 0.75,
  initialImagePath: '',
  maskPath: '',
  shouldFitToWidthHeight: true,
  shouldGenerateVariations: false,
  variationAmount: 0.1,
  seedWeights: '',
  shouldRunESRGAN: false,
  upscalingLevel: 4,
  upscalingStrength: 0.75,
  shouldRunGFPGAN: false,
  gfpganStrength: 0.8,
  shouldRandomizeSeed: true,
};

const initialState: SDState = initialSDState;

export const sdSlice = createSlice({
  name: 'sd',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string>) => {
      state.prompt = action.payload;
    },
    setIterations: (state, action: PayloadAction<number>) => {
      state.iterations = action.payload;
    },
    setSteps: (state, action: PayloadAction<number>) => {
      state.steps = action.payload;
    },
    setCfgScale: (state, action: PayloadAction<number>) => {
      state.cfgScale = action.payload;
    },
    setThreshold: (state, action: PayloadAction<number>) => {
      state.threshold = action.payload;
    },
    setPerlin: (state, action: PayloadAction<number>) => {
      state.perlin = action.payload;
    },
    setHeight: (state, action: PayloadAction<number>) => {
      state.height = action.payload;
    },
    setWidth: (state, action: PayloadAction<number>) => {
      state.width = action.payload;
    },
    setSampler: (state, action: PayloadAction<string>) => {
      state.sampler = action.payload;
    },
    setSeed: (state, action: PayloadAction<number>) => {
      state.seed = action.payload;
      state.shouldRandomizeSeed = false;
    },
    setImg2imgStrength: (state, action: PayloadAction<number>) => {
      state.img2imgStrength = action.payload;
    },
    setGfpganStrength: (state, action: PayloadAction<number>) => {
      state.gfpganStrength = action.payload;
    },
    setUpscalingLevel: (state, action: PayloadAction<UpscalingLevel>) => {
      state.upscalingLevel = action.payload;
    },
    setUpscalingStrength: (state, action: PayloadAction<number>) => {
      state.upscalingStrength = action.payload;
    },
    setShouldUseInitImage: (state, action: PayloadAction<boolean>) => {
      state.shouldUseInitImage = action.payload;
    },
    setInitialImagePath: (state, action: PayloadAction<string>) => {
      const newInitialImagePath = action.payload;
      state.shouldUseInitImage = newInitialImagePath ? true : false;
      state.initialImagePath = newInitialImagePath;
    },
    setMaskPath: (state, action: PayloadAction<string>) => {
      state.maskPath = action.payload;
    },
    setSeamless: (state, action: PayloadAction<boolean>) => {
      state.seamless = action.payload;
    },
    setShouldFitToWidthHeight: (state, action: PayloadAction<boolean>) => {
      state.shouldFitToWidthHeight = action.payload;
    },
    resetSeed: (state) => {
      state.seed = -1;
    },
    setParameter: (
      state,
      action: PayloadAction<{ key: string; value: string | number | boolean }>
    ) => {
      // TODO: This probably needs to be refactored.
      const { key, value } = action.payload;
      const temp = { ...state, [key]: value };
      if (key === 'seed') {
        temp.shouldRandomizeSeed = false;
      }
      if (key === 'initialImagePath' && value === '') {
        temp.shouldUseInitImage = false;
      }
      return temp;
    },
    setShouldGenerateVariations: (state, action: PayloadAction<boolean>) => {
      state.shouldGenerateVariations = action.payload;
    },
    setVariationAmount: (state, action: PayloadAction<number>) => {
      state.variationAmount = action.payload;
    },
    setSeedWeights: (state, action: PayloadAction<string>) => {
      state.seedWeights = action.payload;
    },
    setAllParameters: (state, action: PayloadAction<SDMetadata>) => {
      // TODO: This probably needs to be refactored.
      const {
        prompt,
        steps,
        cfgScale,
        threshold,
        perlin,
        height,
        width,
        sampler,
        seed,
        img2imgStrength,
        gfpganStrength,
        upscalingLevel,
        upscalingStrength,
        initialImagePath,
        maskPath,
        seamless,
        shouldFitToWidthHeight,
      } = action.payload;

      // ?? = falsy values ('', 0, etc) are used
      // || = falsy values not used
      state.prompt = prompt ?? state.prompt;
      state.steps = steps || state.steps;
      state.cfgScale = cfgScale || state.cfgScale;
      state.threshold = threshold || state.threshold;
      state.perlin = perlin || state.perlin;
      state.width = width || state.width;
      state.height = height || state.height;
      state.sampler = sampler || state.sampler;
      state.seed = seed ?? state.seed;
      state.seamless = seamless ?? state.seamless;
      state.shouldFitToWidthHeight =
        shouldFitToWidthHeight ?? state.shouldFitToWidthHeight;
      state.img2imgStrength = img2imgStrength ?? state.img2imgStrength;
      state.gfpganStrength = gfpganStrength ?? state.gfpganStrength;
      state.upscalingLevel = upscalingLevel ?? state.upscalingLevel;
      state.upscalingStrength = upscalingStrength ?? state.upscalingStrength;
      state.initialImagePath = initialImagePath ?? state.initialImagePath;
      state.maskPath = maskPath ?? state.maskPath;

      // If the image whose parameters we are using has a seed, disable randomizing the seed
      if (seed) {
        state.shouldRandomizeSeed = false;
      }

      // if we have a gfpgan strength, enable it
      state.shouldRunGFPGAN = gfpganStrength ? true : false;

      // if we have a esrgan strength, enable it
      state.shouldRunESRGAN = upscalingLevel ? true : false;

      // if we want to recreate an image exactly, we disable variations
      state.shouldGenerateVariations = false;

      state.shouldUseInitImage = initialImagePath ? true : false;
    },
    resetSDState: (state) => {
      return {
        ...state,
        ...initialSDState,
      };
    },
    setShouldRunGFPGAN: (state, action: PayloadAction<boolean>) => {
      state.shouldRunGFPGAN = action.payload;
    },
    setShouldRunESRGAN: (state, action: PayloadAction<boolean>) => {
      state.shouldRunESRGAN = action.payload;
    },
    setShouldRandomizeSeed: (state, action: PayloadAction<boolean>) => {
      state.shouldRandomizeSeed = action.payload;
    },
  },
});

export const {
  setPrompt,
  setIterations,
  setSteps,
  setCfgScale,
  setThreshold,
  setPerlin,
  setHeight,
  setWidth,
  setSampler,
  setSeed,
  setSeamless,
  setImg2imgStrength,
  setGfpganStrength,
  setUpscalingLevel,
  setUpscalingStrength,
  setShouldUseInitImage,
  setInitialImagePath,
  setMaskPath,
  resetSeed,
  resetSDState,
  setShouldFitToWidthHeight,
  setParameter,
  setShouldGenerateVariations,
  setSeedWeights,
  setVariationAmount,
  setAllParameters,
  setShouldRunGFPGAN,
  setShouldRunESRGAN,
  setShouldRandomizeSeed,
} = sdSlice.actions;

export default sdSlice.reducer;
