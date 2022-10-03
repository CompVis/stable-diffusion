type FeatureHelpInfo = {
  text: string;
  href: string;
  guideImage: string;
};

export enum Feature {
  PROMPT,
  GALLERY,
  OTHER,
  SEED,
  VARIATIONS,
  UPSCALE,
  FACE_CORRECTION,
  IMAGE_TO_IMAGE,
}

export const FEATURES: Record<Feature, FeatureHelpInfo> = {
  [Feature.PROMPT]: {
    text: 'This field will take all prompt text, including both content and stylistic terms. CLI Commands will not work in the prompt.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.GALLERY]: {
    text: 'As new invocations are generated, files from the output directory will be displayed here. Generations have additional options to configure new generations.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.OTHER]: {
    text: 'Additional Options',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.SEED]: {
    text: 'Seed values provide an initial set of noise which guide the denoising process.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.VARIATIONS]: {
    text: 'Try a variation with an amount of between 0 and 1 to change the output image for the set seed.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.UPSCALE]: {
    text: 'Using ESRGAN you can increase the output resolution without requiring a higher width/height in the initial generation.',
    href: 'link/to/docs/feature1.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.FACE_CORRECTION]: {
    text: 'Using GFPGAN or CodeFormer, Face Correction will attempt to identify faces in outputs, and correct any defects/abnormalities. Higher values will apply a stronger corrective pressure on outputs.',
    href: 'link/to/docs/feature2.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.IMAGE_TO_IMAGE]: {
    text: 'ImageToImage allows the upload of an initial image, which InvokeAI will use to guide the generation process, along with a prompt. A lower value for this setting will more closely resemble the original image. Values between 0-1 are accepted, and a range of .25-.75 is recommended ',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
};
