import {
  Box,
  Flex,
  Icon,
  IconButton,
  Image,
  Tooltip,
  useColorModeValue,
} from '@chakra-ui/react';
import { useAppDispatch } from '../../app/store';
import { setCurrentImage } from './gallerySlice';
import { FaCheck, FaSeedling, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { memo, SyntheticEvent, useState } from 'react';
import { setAllParameters, setSeed } from '../options/optionsSlice';
import * as InvokeAI from '../../app/invokeai';
import { IoArrowUndoCircleOutline } from 'react-icons/io5';

interface HoverableImageProps {
  image: InvokeAI.Image;
  isSelected: boolean;
}

const memoEqualityCheck = (
  prev: HoverableImageProps,
  next: HoverableImageProps
) => prev.image.uuid === next.image.uuid && prev.isSelected === next.isSelected;

/**
 * Gallery image component with delete/use all/use seed buttons on hover.
 */
const HoverableImage = memo((props: HoverableImageProps) => {
  const [isHovered, setIsHovered] = useState<boolean>(false);
  const dispatch = useAppDispatch();

  const checkColor = useColorModeValue('green.600', 'green.300');
  const bgColor = useColorModeValue('gray.200', 'gray.700');
  const bgGradient = useColorModeValue(
    'radial-gradient(circle, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0.7) 20%, rgba(0,0,0,0) 100%)',
    'radial-gradient(circle, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.7) 20%, rgba(0,0,0,0) 100%)'
  );

  const { image, isSelected } = props;
  const { url, uuid, metadata } = image;

  const handleMouseOver = () => setIsHovered(true);
  const handleMouseOut = () => setIsHovered(false);

  const handleClickSetAllParameters = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setAllParameters(metadata));
  };

  const handleClickSetSeed = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setSeed(image.metadata.image.seed));
  };

  const handleClickImage = () => dispatch(setCurrentImage(image));

  return (
    <Box position={'relative'} key={uuid}>
      <Image
        width={120}
        height={120}
        objectFit="cover"
        rounded={'md'}
        src={url}
        loading={'lazy'}
        backgroundColor={bgColor}
      />
      <Flex
        cursor={'pointer'}
        position={'absolute'}
        top={0}
        left={0}
        rounded={'md'}
        width="100%"
        height="100%"
        alignItems={'center'}
        justifyContent={'center'}
        background={isSelected ? bgGradient : undefined}
        onClick={handleClickImage}
        onMouseOver={handleMouseOver}
        onMouseOut={handleMouseOut}
      >
        {isSelected && (
          <Icon fill={checkColor} width={'50%'} height={'50%'} as={FaCheck} />
        )}
        {isHovered && (
          <Flex
            direction={'column'}
            gap={1}
            position={'absolute'}
            top={1}
            right={1}
          >
            <Tooltip label={'Delete image'}>
              <DeleteImageModal image={image}>
                <IconButton
                  colorScheme="red"
                  aria-label="Delete image"
                  icon={<FaTrashAlt />}
                  size="xs"
                  variant={'imageHoverIconButton'}
                  fontSize={14}
                />
              </DeleteImageModal>
            </Tooltip>
            {['txt2img', 'img2img'].includes(image?.metadata?.image?.type) && (
              <Tooltip label="Use all parameters">
                <IconButton
                  aria-label="Use all parameters"
                  icon={<IoArrowUndoCircleOutline />}
                  size="xs"
                  fontSize={18}
                  variant={'imageHoverIconButton'}
                  onClickCapture={handleClickSetAllParameters}
                />
              </Tooltip>
            )}
            {image?.metadata?.image?.seed && (
              <Tooltip label="Use seed">
                <IconButton
                  aria-label="Use seed"
                  icon={<FaSeedling />}
                  size="xs"
                  fontSize={16}
                  variant={'imageHoverIconButton'}
                  onClickCapture={handleClickSetSeed}
                />
              </Tooltip>
            )}
          </Flex>
        )}
      </Flex>
    </Box>
  );
}, memoEqualityCheck);

export default HoverableImage;
