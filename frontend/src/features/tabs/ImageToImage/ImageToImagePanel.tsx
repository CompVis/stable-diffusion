import { Box } from '@chakra-ui/react';
import React from 'react';
import { Feature } from '../../../app/features';
import { RootState, useAppSelector } from '../../../app/store';
import FaceRestore from '../../options/AdvancedOptions/FaceRestore/FaceRestore';
import FaceRestoreOptions from '../../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageFit from '../../options/AdvancedOptions/ImageToImage/ImageFit';
import ImageToImageStrength from '../../options/AdvancedOptions/ImageToImage/ImageToImageStrength';
import SeedOptions from '../../options/AdvancedOptions/Seed/SeedOptions';
import Upscale from '../../options/AdvancedOptions/Upscale/Upscale';
import UpscaleOptions from '../../options/AdvancedOptions/Upscale/UpscaleOptions';
import Variations from '../../options/AdvancedOptions/Variations/Variations';
import VariationsOptions from '../../options/AdvancedOptions/Variations/VariationsOptions';
import MainAdvancedOptions from '../../options/MainOptions/MainAdvancedOptions';
import MainOptions from '../../options/MainOptions/MainOptions';
import OptionsAccordion from '../../options/OptionsAccordion';
import OutputOptions from '../../options/OutputOptions';
import ProcessButtons from '../../options/ProcessButtons/ProcessButtons';
import PromptInput from '../../options/PromptInput/PromptInput';

export default function ImageToImagePanel() {
  const showAdvancedOptions = useAppSelector(
    (state: RootState) => state.options.showAdvancedOptions
  );

  const imageToImageAccordions = {
    seed: {
      header: (
        <Box flex="1" textAlign="left">
          Seed
        </Box>
      ),
      feature: Feature.SEED,
      options: <SeedOptions />,
    },
    variations: {
      header: <Variations />,
      feature: Feature.VARIATIONS,
      options: <VariationsOptions />,
    },
    face_restore: {
      header: <FaceRestore />,
      feature: Feature.FACE_CORRECTION,
      options: <FaceRestoreOptions />,
    },
    upscale: {
      header: <Upscale />,
      feature: Feature.UPSCALE,
      options: <UpscaleOptions />,
    },
    other: {
      header: (
        <Box flex="1" textAlign="left">
          Other
        </Box>
      ),
      feature: Feature.OTHER,
      options: <OutputOptions />,
    },
  };

  return (
    <div className="image-to-image-panel">
      <PromptInput />
      <ProcessButtons />
      <MainOptions />
      <ImageToImageStrength
        label="Image To Image Strength"
        styleClass="main-option-block image-to-image-strength-main-option"
      />
      <ImageFit />
      <MainAdvancedOptions />
      {showAdvancedOptions ? (
        <OptionsAccordion accordionInfo={imageToImageAccordions} />
      ) : null}
    </div>
  );
}
