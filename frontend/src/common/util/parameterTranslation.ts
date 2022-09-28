/*
    These functions translate frontend state into parameters
    suitable for consumption by the backend, and vice-versa.
*/

import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from '../../app/constants';
import { OptionsState } from '../../features/options/optionsSlice';
import { SystemState } from '../../features/system/systemSlice';
import {
  seedWeightsToString,
  stringToSeedWeightsArray,
} from './seedWeightPairs';
import randomInt from './randomInt';

export const frontendToBackendParameters = (
  optionsState: OptionsState,
  systemState: SystemState
): { [key: string]: any } => {
  const {
    prompt,
    iterations,
    steps,
    cfgScale,
    height,
    width,
    sampler,
    seed,
    seamless,
    shouldUseInitImage,
    img2imgStrength,
    initialImagePath,
    maskPath,
    shouldFitToWidthHeight,
    shouldGenerateVariations,
    variationAmount,
    seedWeights,
    shouldRunESRGAN,
    upscalingLevel,
    upscalingStrength,
    shouldRunGFPGAN,
    gfpganStrength,
    shouldRandomizeSeed,
  } = optionsState;

  const { shouldDisplayInProgress } = systemState;

  const generationParameters: { [k: string]: any } = {
    prompt,
    iterations,
    steps,
    cfg_scale: cfgScale,
    height,
    width,
    sampler_name: sampler,
    seed,
    seamless,
    progress_images: shouldDisplayInProgress,
  };

  generationParameters.seed = shouldRandomizeSeed
    ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
    : seed;

  if (shouldUseInitImage) {
    generationParameters.init_img = initialImagePath;
    generationParameters.strength = img2imgStrength;
    generationParameters.fit = shouldFitToWidthHeight;
    if (maskPath) {
      generationParameters.init_mask = maskPath;
    }
  }

  if (shouldGenerateVariations) {
    generationParameters.variation_amount = variationAmount;
    if (seedWeights) {
      generationParameters.with_variations =
        stringToSeedWeightsArray(seedWeights);
    }
  } else {
    generationParameters.variation_amount = 0;
  }

  let esrganParameters: false | { [k: string]: any } = false;
  let gfpganParameters: false | { [k: string]: any } = false;

  if (shouldRunESRGAN) {
    esrganParameters = {
      level: upscalingLevel,
      strength: upscalingStrength,
    };
  }

  if (shouldRunGFPGAN) {
    gfpganParameters = {
      strength: gfpganStrength,
    };
  }

  return {
    generationParameters,
    esrganParameters,
    gfpganParameters,
  };
};

export const backendToFrontendParameters = (parameters: {
  [key: string]: any;
}) => {
  const {
    prompt,
    iterations,
    steps,
    cfg_scale,
    height,
    width,
    sampler_name,
    seed,
    seamless,
    progress_images,
    variation_amount,
    with_variations,
    gfpgan_strength,
    upscale,
    init_img,
    init_mask,
    strength,
  } = parameters;

  const options: { [key: string]: any } = {
    shouldDisplayInProgress: progress_images,
    // init
    shouldGenerateVariations: false,
    shouldRunESRGAN: false,
    shouldRunGFPGAN: false,
    initialImagePath: '',
    maskPath: '',
  };

  if (variation_amount > 0) {
    options.shouldGenerateVariations = true;
    options.variationAmount = variation_amount;
    if (with_variations) {
      options.seedWeights = seedWeightsToString(with_variations);
    }
  }

  if (gfpgan_strength > 0) {
    options.shouldRunGFPGAN = true;
    options.gfpganStrength = gfpgan_strength;
  }

  if (upscale) {
    options.shouldRunESRGAN = true;
    options.upscalingLevel = upscale[0];
    options.upscalingStrength = upscale[1];
  }

  if (init_img) {
    options.shouldUseInitImage = true;
    options.initialImagePath = init_img;
    options.strength = strength;
    if (init_mask) {
      options.maskPath = init_mask;
    }
  }

  // if we had a prompt, add all the metadata, but if we don't have a prompt,
  // we must have only done ESRGAN or GFPGAN so do not add that metadata
  if (prompt) {
    options.prompt = prompt;
    options.iterations = iterations;
    options.steps = steps;
    options.cfgScale = cfg_scale;
    options.height = height;
    options.width = width;
    options.sampler = sampler_name;
    options.seed = seed;
    options.seamless = seamless;
  }

  return options;
};
