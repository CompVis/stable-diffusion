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
/** For each tooltip in the UI, the below feature definitions & props will pull relevant information into the tooltip.
 * 
 * To-do: href & GuideImages are placeholders, and are not currently utilized, but will be updated (along with the tooltip UI) as feature and UI development and we get a better idea on where things "forever homes" will be . 
 */
export const FEATURES: Record<Feature, FeatureHelpInfo> = {
  [Feature.PROMPT]: {
    text: 'This field will take all prompt text, including both content and stylistic terms. While weights can be included in the prompt, standard CLI Commands/parameters will not work.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.GALLERY]: {
    text: 'As new invocations are generated, files from the output directory will be displayed here. Generations have additional options to configure new generations.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.OTHER]: {
    text: 'These options will enable alternative processing modes for Invoke. Seamless tiling will work to generate repeating patterns in the output. High Resolution Optimization performs a two-step generation cycle, and should be used at higher resolutions when you desire a more coherent image/composition. ',    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.SEED]: {
    text: 'Seed values provide an initial set of noise which guide the denoising process, and can be randomized or populated with a seed from a previous invocation. The Threshold feature can be used to mitigate undesirable outcomes at higher CFG values (try between 0-10), and Perlin can be used to add Perlin noise into the denoising process - Both serve to add variation to your outputs. ',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.VARIATIONS]: {
    text: 'Try a variation with an amount of between 0 and 1 to change the output image for the set seed - Interesting variations on the seed are found between 0.1 and 0.3.',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.UPSCALE]: {
    text: 'Using ESRGAN you can increase the output resolution without requiring a higher width/height in the initial generation.',
    href: 'link/to/docs/feature1.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.FACE_CORRECTION]: {
    text: 'Using GFPGAN, Face Correction will attempt to identify faces in outputs, and correct any defects/abnormalities. Higher values will apply a stronger corrective pressure on outputs, resulting in more appealing faces (with less respect for accuracy of the original subject).',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
  [Feature.IMAGE_TO_IMAGE]: {
    text: 'ImageToImage allows the upload of an initial image, which InvokeAI will use to guide the generation process, along with a prompt. A lower value for this setting will more closely resemble the original image. Values between 0-1 are accepted, and a range of .25-.75 is recommended ',
    href: 'link/to/docs/feature3.html',
    guideImage: 'asset/path.gif',
  },
};
