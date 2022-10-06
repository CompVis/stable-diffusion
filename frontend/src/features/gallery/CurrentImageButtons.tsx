import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import * as InvokeAI from '../../app/invokeai';

import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import {
  setActiveTab,
  setAllParameters,
  setInitialImagePath,
  setSeed,
  setShouldShowImageDetails,
} from '../options/optionsSlice';
import DeleteImageModal from './DeleteImageModal';
import { SystemState } from '../system/systemSlice';
import IAIButton from '../../common/components/IAIButton';
import { runESRGAN, runGFPGAN } from '../../app/socketio/actions';
import IAIIconButton from '../../common/components/IAIIconButton';
import { MdDelete, MdFace, MdHd, MdImage, MdInfo } from 'react-icons/md';
import InvokePopover from './InvokePopover';
import UpscaleOptions from '../options/AdvancedOptions/Upscale/UpscaleOptions';
import FaceRestoreOptions from '../options/AdvancedOptions/FaceRestore/FaceRestoreOptions';
import { useHotkeys } from 'react-hotkeys-hook';
import { useToast } from '@chakra-ui/react';

const systemSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    return {
      isProcessing: system.isProcessing,
      isConnected: system.isConnected,
      isGFPGANAvailable: system.isGFPGANAvailable,
      isESRGANAvailable: system.isESRGANAvailable,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type CurrentImageButtonsProps = {
  image: InvokeAI.Image;
};

/**
 * Row of buttons for common actions:
 * Use as init image, use all params, use seed, upscale, fix faces, details, delete.
 */
const CurrentImageButtons = ({ image }: CurrentImageButtonsProps) => {
  const dispatch = useAppDispatch();

  const shouldShowImageDetails = useAppSelector(
    (state: RootState) => state.options.shouldShowImageDetails
  );

  const toast = useToast();

  const intermediateImage = useAppSelector(
    (state: RootState) => state.gallery.intermediateImage
  );

  const upscalingLevel = useAppSelector(
    (state: RootState) => state.options.upscalingLevel
  );

  const gfpganStrength = useAppSelector(
    (state: RootState) => state.options.gfpganStrength
  );

  const { isProcessing, isConnected, isGFPGANAvailable, isESRGANAvailable } =
    useAppSelector(systemSelector);

  const handleClickUseAsInitialImage = () => {
    dispatch(setInitialImagePath(image.url));
    dispatch(setActiveTab(1));
  };

  useHotkeys(
    'shift+i',
    () => {
      if (image) {
        handleClickUseAsInitialImage();
        toast({
          title: 'Sent To Image To Image',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'No Image Loaded',
          description: 'No image found to send to image to image module.',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [image]
  );

  const handleClickUseAllParameters = () =>
    dispatch(setAllParameters(image.metadata));
  useHotkeys(
    'a',
    () => {
      if (['txt2img', 'img2img'].includes(image?.metadata?.image?.type)) {
        handleClickUseAllParameters();
        toast({
          title: 'Parameters Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Parameters Not Set',
          description: 'No metadata found for this image.',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [image]
  );

  // Non-null assertion: this button is disabled if there is no seed.
  // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
  const handleClickUseSeed = () => dispatch(setSeed(image.metadata.image.seed));
  useHotkeys(
    's',
    () => {
      if (image?.metadata?.image?.seed) {
        handleClickUseSeed();
        toast({
          title: 'Seed Set',
          status: 'success',
          duration: 2500,
          isClosable: true,
        });
      } else {
        toast({
          title: 'Seed Not Set',
          description: 'Could not find seed for this image.',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [image]
  );

  const handleClickUpscale = () => dispatch(runESRGAN(image));
  useHotkeys(
    'u',
    () => {
      if (
        isESRGANAvailable &&
        Boolean(!intermediateImage) &&
        isConnected &&
        !isProcessing &&
        upscalingLevel
      ) {
        handleClickUpscale();
      } else {
        toast({
          title: 'Upscaling Failed',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [
      image,
      isESRGANAvailable,
      intermediateImage,
      isConnected,
      isProcessing,
      upscalingLevel,
    ]
  );

  const handleClickFixFaces = () => dispatch(runGFPGAN(image));
  useHotkeys(
    'r',
    () => {
      if (
        isGFPGANAvailable &&
        Boolean(!intermediateImage) &&
        isConnected &&
        !isProcessing &&
        gfpganStrength
      ) {
        handleClickFixFaces();
      } else {
        toast({
          title: 'Face Restoration Failed',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [
      image,
      isGFPGANAvailable,
      intermediateImage,
      isConnected,
      isProcessing,
      gfpganStrength,
    ]
  );

  const handleClickShowImageDetails = () =>
    dispatch(setShouldShowImageDetails(!shouldShowImageDetails));

  useHotkeys(
    'i',
    () => {
      if (image) {
        handleClickShowImageDetails();
      } else {
        toast({
          title: 'Failed to load metadata',
          status: 'error',
          duration: 2500,
          isClosable: true,
        });
      }
    },
    [image, shouldShowImageDetails]
  );

  return (
    <div className="current-image-options">
      <IAIIconButton
        icon={<MdImage />}
        tooltip="Send To Image To Image"
        aria-label="Send To Image To Image"
        onClick={handleClickUseAsInitialImage}
      />

      <IAIButton
        label="Use All"
        isDisabled={
          !['txt2img', 'img2img'].includes(image?.metadata?.image?.type)
        }
        onClick={handleClickUseAllParameters}
      />

      <IAIButton
        label="Use Seed"
        isDisabled={!image?.metadata?.image?.seed}
        onClick={handleClickUseSeed}
      />

      <InvokePopover
        title="Restore Faces"
        popoverOptions={<FaceRestoreOptions />}
        actionButton={
          <IAIButton
            label={'Restore Faces'}
            isDisabled={
              !isGFPGANAvailable ||
              Boolean(intermediateImage) ||
              !(isConnected && !isProcessing) ||
              !gfpganStrength
            }
            onClick={handleClickFixFaces}
          />
        }
      >
        <IAIIconButton icon={<MdFace />} aria-label="Restore Faces" />
      </InvokePopover>

      <InvokePopover
        title="Upscale"
        styleClass="upscale-popover"
        popoverOptions={<UpscaleOptions />}
        actionButton={
          <IAIButton
            label={'Upscale Image'}
            isDisabled={
              !isESRGANAvailable ||
              Boolean(intermediateImage) ||
              !(isConnected && !isProcessing) ||
              !upscalingLevel
            }
            onClick={handleClickUpscale}
          />
        }
      >
        <IAIIconButton icon={<MdHd />} aria-label="Upscale" />
      </InvokePopover>

      <IAIIconButton
        icon={<MdInfo />}
        tooltip="Details"
        aria-label="Details"
        onClick={handleClickShowImageDetails}
      />

      <DeleteImageModal image={image}>
        <IAIIconButton
          icon={<MdDelete />}
          tooltip="Delete Image"
          aria-label="Delete Image"
          isDisabled={Boolean(intermediateImage)}
        />
      </DeleteImageModal>
    </div>
  );
};

export default CurrentImageButtons;
