import { Button, Flex, IconButton, useToast } from '@chakra-ui/react';
import { SyntheticEvent, useCallback } from 'react';
import { FaTrash, FaUpload } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { OptionsState, setInitialImagePath, setMaskPath } from './optionsSlice';
import {
  uploadInitialImage,
  uploadMaskImage,
} from '../../app/socketio/actions';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import ImageUploader from './ImageUploader';
import { FileRejection } from 'react-dropzone';

const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      initialImagePath: options.initialImagePath,
      maskPath: options.maskPath,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

type InitAndMaskUploadButtonsProps = {
  setShouldShowMask: (b: boolean) => void;
};

/**
 * Init and mask image upload buttons.
 */
const InitAndMaskUploadButtons = ({
  setShouldShowMask,
}: InitAndMaskUploadButtonsProps) => {
  const dispatch = useAppDispatch();
  const { initialImagePath, maskPath } = useAppSelector(optionsSelector);

  // Use a toast to alert user when a file upload is rejected
  const toast = useToast();

  // Clear the init and mask images
  const handleClickResetInitialImage = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setInitialImagePath(''));
  };

  // Clear the init and mask images
  const handleClickResetMask = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setMaskPath(''));
  };

  // Handle hover to view initial image and mask image
  const handleMouseOverInitialImageUploadButton = () =>
    setShouldShowMask(false);
  const handleMouseOutInitialImageUploadButton = () => setShouldShowMask(true);

  const handleMouseOverMaskUploadButton = () => setShouldShowMask(true);
  const handleMouseOutMaskUploadButton = () => setShouldShowMask(true);

  // Callbacks to for handling file upload attempts
  const initImageFileAcceptedCallback = useCallback(
    (file: File) => dispatch(uploadInitialImage(file)),
    [dispatch]
  );

  const maskImageFileAcceptedCallback = useCallback(
    (file: File) => dispatch(uploadMaskImage(file)),
    [dispatch]
  );

  const fileRejectionCallback = useCallback(
    (rejection: FileRejection) => {
      const msg = rejection.errors.reduce(
        (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
        ''
      );

      toast({
        title: 'Upload failed',
        description: msg,
        status: 'error',
        isClosable: true,
      });
    },
    [toast]
  );

  return (
    <Flex gap={2} justifyContent={'space-between'} width={'100%'}>
      <ImageUploader
        fileAcceptedCallback={initImageFileAcceptedCallback}
        fileRejectionCallback={fileRejectionCallback}
      >
        <Button
          size={'sm'}
          fontSize={'md'}
          fontWeight={'normal'}
          onMouseOver={handleMouseOverInitialImageUploadButton}
          onMouseOut={handleMouseOutInitialImageUploadButton}
          leftIcon={<FaUpload />}
          width={'100%'}
        >
          Image
        </Button>
      </ImageUploader>

      <IconButton
        isDisabled={!initialImagePath}
        size={'sm'}
        aria-label={'Reset mask'}
        onClick={handleClickResetInitialImage}
        icon={<FaTrash />}
      />

      <ImageUploader
        fileAcceptedCallback={maskImageFileAcceptedCallback}
        fileRejectionCallback={fileRejectionCallback}
      >
        <Button
          isDisabled={!initialImagePath}
          size={'sm'}
          fontSize={'md'}
          fontWeight={'normal'}
          onMouseOver={handleMouseOverMaskUploadButton}
          onMouseOut={handleMouseOutMaskUploadButton}
          leftIcon={<FaUpload />}
          width={'100%'}
        >
          Mask
        </Button>
      </ImageUploader>

      <IconButton
        isDisabled={!maskPath}
        size={'sm'}
        aria-label={'Reset mask'}
        onClick={handleClickResetMask}
        icon={<FaTrash />}
      />
    </Flex>
  );
};

export default InitAndMaskUploadButtons;
