import {
  Box,
  Accordion,
  ExpandedIndex,
  // ExpandedIndex,
} from '@chakra-ui/react';

// import { RootState } from '../../app/store';
// import { useAppDispatch, useAppSelector } from '../../app/store';

// import { setOpenAccordions } from '../system/systemSlice';

import OutputOptions from './OutputOptions';
import ImageToImageOptions from './AdvancedOptions/ImageToImage/ImageToImageOptions';
import { Feature } from '../../app/features';
import SeedOptions from './AdvancedOptions/Seed/SeedOptions';
import Upscale from './AdvancedOptions/Upscale/Upscale';
import UpscaleOptions from './AdvancedOptions/Upscale/UpscaleOptions';
import FaceRestore from './AdvancedOptions/FaceRestore/FaceRestore';
import FaceRestoreOptions from './AdvancedOptions/FaceRestore/FaceRestoreOptions';
import ImageToImage from './AdvancedOptions/ImageToImage/ImageToImage';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { setOpenAccordions } from '../system/systemSlice';
import InvokeAccordionItem from './AccordionItems/InvokeAccordionItem';
import Variations from './AdvancedOptions/Variations/Variations';
import VariationsOptions from './AdvancedOptions/Variations/VariationsOptions';

/**
 * Main container for generation and processing parameters.
 */
const OptionsAccordion = () => {
  const openAccordions = useAppSelector(
    (state: RootState) => state.system.openAccordions
  );

  const dispatch = useAppDispatch();

  /**
   * Stores accordion state in redux so preferred UI setup is retained.
   */
  const handleChangeAccordionState = (openAccordions: ExpandedIndex) =>
    dispatch(setOpenAccordions(openAccordions));

  return (
    <Accordion
      defaultIndex={openAccordions}
      allowMultiple
      reduceMotion
      onChange={handleChangeAccordionState}
      className="advanced-settings"
    >
      <InvokeAccordionItem
        header={
          <Box flex="1" textAlign="left">
            Seed
          </Box>
        }
        feature={Feature.SEED}
        options={<SeedOptions />}
      />

      <InvokeAccordionItem
        header={<Variations />}
        feature={Feature.VARIATIONS}
        options={<VariationsOptions />}
      />

      <InvokeAccordionItem
        header={<FaceRestore />}
        feature={Feature.FACE_CORRECTION}
        options={<FaceRestoreOptions />}
      />

      <InvokeAccordionItem
        header={<Upscale />}
        feature={Feature.UPSCALE}
        options={<UpscaleOptions />}
      />

      <InvokeAccordionItem
        header={<ImageToImage />}
        feature={Feature.IMAGE_TO_IMAGE}
        options={<ImageToImageOptions />}
      />

      <InvokeAccordionItem
        header={
          <Box flex="1" textAlign="left">
            Other
          </Box>
        }
        feature={Feature.OTHER}
        options={<OutputOptions />}
      />
    </Accordion>
  );
};

export default OptionsAccordion;
