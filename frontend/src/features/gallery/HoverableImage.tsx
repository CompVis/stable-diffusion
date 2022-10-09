import { Box, Icon, IconButton, Image, Tooltip } from '@chakra-ui/react';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { setCurrentImage } from './gallerySlice';
import { FaCheck, FaImage, FaSeedling, FaTrashAlt } from 'react-icons/fa';
import DeleteImageModal from './DeleteImageModal';
import { memo, SyntheticEvent, useState } from 'react';
import {
  setActiveTab,
  setAllParameters,
  setInitialImagePath,
  setSeed,
} from '../options/optionsSlice';
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

  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
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

  const handleSetInitImage = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setInitialImagePath(image.url));
    if (activeTab !== 1) {
      dispatch(setActiveTab(1));
    }
  };

  const handleClickImage = () => dispatch(setCurrentImage(image));

  return (
    <Box
      position={'relative'}
      key={uuid}
      className="hoverable-image"
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
    >
      <Image
        objectFit="cover"
        rounded={'md'}
        src={url}
        loading={'lazy'}
        className="hoverable-image-image"
      />
      <div className="hoverable-image-content" onClick={handleClickImage}>
        {isSelected && (
          <Icon
            width={'50%'}
            height={'50%'}
            as={FaCheck}
            className="hoverable-image-check"
          />
        )}
      </div>
      {isHovered && (
        <div className="hoverable-image-icons">
          <Tooltip label={'Delete image'} hasArrow>
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
            <Tooltip label="Use All Parameters" hasArrow>
              <IconButton
                aria-label="Use All Parameters"
                icon={<IoArrowUndoCircleOutline />}
                size="xs"
                fontSize={18}
                variant={'imageHoverIconButton'}
                onClickCapture={handleClickSetAllParameters}
              />
            </Tooltip>
          )}
          {image?.metadata?.image?.seed !== undefined && (
            <Tooltip label="Use Seed" hasArrow>
              <IconButton
                aria-label="Use Seed"
                icon={<FaSeedling />}
                size="xs"
                fontSize={16}
                variant={'imageHoverIconButton'}
                onClickCapture={handleClickSetSeed}
              />
            </Tooltip>
          )}
          <Tooltip label="Send To Image To Image" hasArrow>
            <IconButton
              aria-label="Send To Image To Image"
              icon={<FaImage />}
              size="xs"
              fontSize={16}
              variant={'imageHoverIconButton'}
              onClickCapture={handleSetInitImage}
            />
          </Tooltip>
        </div>
      )}
    </Box>
  );
}, memoEqualityCheck);

export default HoverableImage;
