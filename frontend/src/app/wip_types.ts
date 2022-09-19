
/**
 * Defines common parameters required to generate an image.
 * See #266 for the eventual maturation of this interface.
 */
interface CommonParameters {
  /**
   * The "txt2img" prompt. String. Minimum one character. No maximum.
   */
  prompt: string;
  /**
   * The number of sampler steps. Integer. Minimum value 1. No maximum.
   */
  steps: number;
  /**
   * Classifier-free guidance scale. Float. Minimum value 0. Maximum?
   */
  cfgScale: number;
  /**
   * Height of output image in pixels. Integer. Minimum 64. Must be multiple of 64. No maximum.
   */
  height: number;
  /**
   * Width of output image in pixels. Integer. Minimum 64. Must be multiple of 64. No maximum.
   */
  width: number;
  /**
   * Name of the sampler to use. String. Restricted values.
   */
  sampler:
    | 'ddim'
    | 'plms'
    | 'k_lms'
    | 'k_dpm_2'
    | 'k_dpm_2_a'
    | 'k_euler'
    | 'k_euler_a'
    | 'k_heun';
  /**
   * Seed used for randomness. Integer. 0 --> 4294967295, inclusive.
   */
  seed: number;
  /**
   * Flag to enable seamless tiling image generation. Boolean.
   */
  seamless: boolean;
}

/**
 * Defines parameters needed to use the "img2img" generation method.
 */
interface ImageToImageParameters {
  /**
   * Folder path to the image used as the initial image. String.
   */
  initialImagePath: string;
  /**
   * Flag to enable the use of a mask image during "img2img" generations.
   * Requires valid ImageToImageParameters. Boolean.
   */
  shouldUseMaskImage: boolean;
  /**
   * Folder path to the image used as a mask image. String.
   */
  maskImagePath: string;
  /**
   * Strength of adherance to initial image. Float. 0 --> 1, exclusive.
   */
  img2imgStrength: number;
  /**
   * Flag to enable the stretching of init image to desired output. Boolean.
   */
  shouldFit: boolean;
}

/**
 * Defines the parameters needed to generate variations.
 */
interface VariationParameters {
  /**
   * Variation amount. Float. 0 --> 1, exclusive.
   * TODO: What does this really do?
   */
  variationAmount: number;
  /**
   * List of seed-weight pairs formatted as "seed:weight,...".
   * Seed is a valid seed. Weight is a float, 0 --> 1, exclusive.
   * String, must be parseable into [[seed,weight],...] format.
   */
  seedWeights: string;
}

/**
 * Defines the parameters needed to use GFPGAN postprocessing.
 */
interface GFPGANParameters {
  /**
   * GFPGAN strength. Strength to apply face-fixing processing. Float. 0 --> 1, exclusive.
   */
  gfpganStrength: number;
}

/**
 * Defines the parameters needed to use ESRGAN postprocessing.
 */
interface ESRGANParameters {
  /**
   * ESRGAN strength. Strength to apply upscaling. Float. 0 --> 1, exclusive.
   */
  esrganStrength: number;
  /**
   * ESRGAN upscaling scale. One of 2x | 4x. Represented as integer.
   */
  esrganScale: 2 | 4;
}

/**
 * Extends the generation and processing method parameters, adding flags to enable each.
 */
interface ProcessingParameters extends CommonParameters {
  /**
   * Flag to enable the generation of variations. Requires valid VariationParameters. Boolean.
   */
  shouldGenerateVariations: boolean;
  /**
   * Variation parameters.
   */
  variationParameters: VariationParameters;
  /**
   * Flag to enable the use of an initial image, i.e. to use "img2img" generation.
   * Requires valid ImageToImageParameters. Boolean.
   */
  shouldUseImageToImage: boolean;
  /**
   * ImageToImage parameters.
   */
  imageToImageParameters: ImageToImageParameters;
  /**
   * Flag to enable GFPGAN postprocessing. Requires valid GFPGANParameters. Boolean.
   */
  shouldRunGFPGAN: boolean;
  /**
   * GFPGAN parameters.
   */
  gfpganParameters: GFPGANParameters;
  /**
   * Flag to enable ESRGAN postprocessing. Requires valid ESRGANParameters. Boolean.
   */
  shouldRunESRGAN: boolean;
  /**
   * ESRGAN parameters.
   */
  esrganParameters: GFPGANParameters;
}

/**
 * Extends ProcessingParameters, adding items needed to request processing.
 */
interface ProcessingState extends ProcessingParameters {
  /**
   * Number of images to generate. Integer. Minimum 1.
   */
  iterations: number;
  /**
   * Flag to enable the randomization of the seed on each generation. Boolean.
   */
  shouldRandomizeSeed: boolean;
}


export {}
