import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import * as InvokeAI from '../../app/invokeai';
import promptToString from '../../common/util/promptToString';
import { seedWeightsToString } from '../../common/util/seedWeightPairs';

export type UpscalingLevel = 2 | 4;

export interface OptionsState {
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
  initialImagePath: string | null;
  maskPath: string;
  seamless: boolean;
  shouldFitToWidthHeight: boolean;
  shouldGenerateVariations: boolean;
  variationAmount: number;
  seedWeights: string;
  shouldRunESRGAN: boolean;
  shouldRunGFPGAN: boolean;
  shouldRandomizeSeed: boolean;
  showAdvancedOptions: boolean;
  activeTab: number;
  shouldShowImageDetails: boolean;
  shouldShowGallery: boolean;
}

const initialOptionsState: OptionsState = {
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
  initialImagePath: null,
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
  showAdvancedOptions: true,
  activeTab: 0,
  shouldShowImageDetails: false,
  shouldShowGallery: false,
};

const initialState: OptionsState = initialOptionsState;

export const optionsSlice = createSlice({
  name: 'options',
  initialState,
  reducers: {
    setPrompt: (state, action: PayloadAction<string | InvokeAI.Prompt>) => {
      const newPrompt = action.payload;
      if (typeof newPrompt === 'string') {
        state.prompt = newPrompt;
      } else {
        state.prompt = promptToString(newPrompt);
      }
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
    setInitialImagePath: (state, action: PayloadAction<string | null>) => {
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
    setAllParameters: (state, action: PayloadAction<InvokeAI.Metadata>) => {
      const {
        type,
        sampler,
        prompt,
        seed,
        variations,
        steps,
        cfg_scale,
        threshold,
        perlin,
        seamless,
        width,
        height,
        strength,
        fit,
        init_image_path,
        mask_image_path,
      } = action.payload.image;

      if (type === 'img2img') {
        if (init_image_path) state.initialImagePath = init_image_path;
        if (mask_image_path) state.maskPath = mask_image_path;
        if (strength) state.img2imgStrength = strength;
        if (typeof fit === 'boolean') state.shouldFitToWidthHeight = fit;
        state.shouldUseInitImage = true;
      } else {
        state.shouldUseInitImage = false;
      }

      if (variations && variations.length > 0) {
        state.seedWeights = seedWeightsToString(variations);
        state.shouldGenerateVariations = true;
      } else {
        state.shouldGenerateVariations = false;
      }

      if (seed) {
        state.seed = seed;
        state.shouldRandomizeSeed = false;
      }

      /**
       * We support arbitrary numbers of postprocessing steps, so it
       * doesnt make sense to be include postprocessing metadata when
       * we use all parameters. Because this code needed a bit of braining
       * to figure out, I am leaving it, in case it is needed again.
       */

      // let postprocessingNotDone = ['gfpgan', 'esrgan'];
      // if (postprocessing && postprocessing.length > 0) {
      //   postprocessing.forEach(
      //     (postprocess: InvokeAI.PostProcessedImageMetadata) => {
      //       if (postprocess.type === 'gfpgan') {
      //         const { strength } = postprocess;
      //         if (strength) state.gfpganStrength = strength;
      //         state.shouldRunGFPGAN = true;
      //         postprocessingNotDone = postprocessingNotDone.filter(
      //           (p) => p !== 'gfpgan'
      //         );
      //       }
      //       if (postprocess.type === 'esrgan') {
      //         const { scale, strength } = postprocess;
      //         if (scale) state.upscalingLevel = scale;
      //         if (strength) state.upscalingStrength = strength;
      //         state.shouldRunESRGAN = true;
      //         postprocessingNotDone = postprocessingNotDone.filter(
      //           (p) => p !== 'esrgan'
      //         );
      //       }
      //     }
      //   );
      // }

      // postprocessingNotDone.forEach((p) => {
      //   if (p === 'esrgan') state.shouldRunESRGAN = false;
      //   if (p === 'gfpgan') state.shouldRunGFPGAN = false;
      // });

      if (prompt) state.prompt = promptToString(prompt);
      if (sampler) state.sampler = sampler;
      if (steps) state.steps = steps;
      if (cfg_scale) state.cfgScale = cfg_scale;
      if (threshold) state.threshold = threshold;
      if (typeof threshold === 'undefined') state.threshold = 0;
      if (perlin) state.perlin = perlin;
      if (typeof perlin === 'undefined') state.perlin = 0;      
      if (typeof seamless === 'boolean') state.seamless = seamless;
      if (width) state.width = width;
      if (height) state.height = height;
    },
    resetOptionsState: (state) => {
      return {
        ...state,
        ...initialOptionsState,
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
    setShowAdvancedOptions: (state, action: PayloadAction<boolean>) => {
      state.showAdvancedOptions = action.payload;
    },
    setActiveTab: (state, action: PayloadAction<number>) => {
      state.activeTab = action.payload;
    },
    setShouldShowImageDetails: (state, action: PayloadAction<boolean>) => {
      state.shouldShowImageDetails = action.payload;
    },
    setShouldShowGallery: (state, action: PayloadAction<boolean>) => {
      state.shouldShowGallery = action.payload;
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
  resetOptionsState,
  setShouldFitToWidthHeight,
  setParameter,
  setShouldGenerateVariations,
  setSeedWeights,
  setVariationAmount,
  setAllParameters,
  setShouldRunGFPGAN,
  setShouldRunESRGAN,
  setShouldRandomizeSeed,
  setShowAdvancedOptions,
  setActiveTab,
  setShouldShowImageDetails,
  setShouldShowGallery,
} = optionsSlice.actions;

export default optionsSlice.reducer;
