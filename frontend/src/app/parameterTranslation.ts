import { SDState } from '../features/sd/sdSlice';
import randomInt from '../features/sd/util/randomInt';
import {
    seedWeightsToString,
    stringToSeedWeights,
} from '../features/sd/util/seedWeightPairs';
import { SystemState } from '../features/system/systemSlice';
import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from './constants';

/*
    These functions translate frontend state into parameters
    suitable for consumption by the backend, and vice-versa.
*/

export const frontendToBackendParameters = (
    sdState: SDState,
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
        variantAmount,
        seedWeights,
        shouldRunESRGAN,
        upscalingLevel,
        upscalingStrength,
        shouldRunGFPGAN,
        gfpganStrength,
        shouldRandomizeSeed,
    } = sdState;

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
        generationParameters.variation_amount = variantAmount;
        if (seedWeights) {
            generationParameters.with_variations =
                stringToSeedWeights(seedWeights);
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

    const sd: { [key: string]: any } = {
        shouldDisplayInProgress: progress_images,
        // init
        shouldGenerateVariations: false,
        shouldRunESRGAN: false,
        shouldRunGFPGAN: false,
        initialImagePath: '',
        maskPath: '',
    };

    if (variation_amount > 0) {
        sd.shouldGenerateVariations = true;
        sd.variantAmount = variation_amount;
        if (with_variations) {
            sd.seedWeights = seedWeightsToString(with_variations);
        }
    }

    if (gfpgan_strength > 0) {
        sd.shouldRunGFPGAN = true;
        sd.gfpganStrength = gfpgan_strength;
    }

    if (upscale) {
        sd.shouldRunESRGAN = true;
        sd.upscalingLevel = upscale[0];
        sd.upscalingStrength = upscale[1];
    }

    if (init_img) {
        sd.shouldUseInitImage = true
        sd.initialImagePath = init_img;
        sd.strength = strength;
        if (init_mask) {
            sd.maskPath = init_mask;
        }
    }

    // if we had a prompt, add all the metadata, but if we don't have a prompt,
    // we must have only done ESRGAN or GFPGAN so do not add that metadata
    if (prompt) {
        sd.prompt = prompt;
        sd.iterations = iterations;
        sd.steps = steps;
        sd.cfgScale = cfg_scale;
        sd.height = height;
        sd.width = width;
        sd.sampler = sampler_name;
        sd.seed = seed;
        sd.seamless = seamless;
    }

    return sd;
};
