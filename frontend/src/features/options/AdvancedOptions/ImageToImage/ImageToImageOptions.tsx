import { Flex } from '@chakra-ui/react';
import InitAndMaskImage from '../../InitAndMaskImage';
import ImageFit from './ImageFit';
import ImageToImageStrength from './ImageToImageStrength';

/**
 * Options for img2img generation (strength, fit, init/mask upload).
 */
const ImageToImageOptions = () => {
  return (
    <Flex direction={'column'} gap={2}>
      <ImageToImageStrength />
      <ImageFit />
      <InitAndMaskImage />
    </Flex>
  );
};

export default ImageToImageOptions;
