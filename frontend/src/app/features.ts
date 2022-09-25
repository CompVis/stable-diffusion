type FeatureHelpInfo = {
  text: string;
  href: string;
  guideImage: string;
};

export enum Feature {
  PROMPT,
  GALLERY,
  OUTPUT,
  SEED_AND_VARIATION,
  ESRGAN,
  FACE_CORRECTION,
  IMAGE_TO_IMAGE,
  SAMPLER,
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
  [Feature.OUTPUT]: {
    text: 'The Height and Width of generations can be controlled here. If you experience errors, you may be generating an image too large for your system. The seamless option will more often result in repeating patterns in outputs.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.SEED_AND_VARIATION]: {
    text: 'Seed values provide an initial set of noise which guide the denoising process. Try a variation with an amount of between 0 and 1 to change the output image for that seed.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.ESRGAN]: {
    text: 'The ESRGAN setting can be used to increase the output resolution without requiring a higher width/height in the initial generation.',
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
  [Feature.SAMPLER]: {
    text: 'This setting allows for different denoising samplers to be used, as well as the number of denoising steps used, which will change the resulting output.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
};
