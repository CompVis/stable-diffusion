import {
  Button,
  Flex,
  IconButton,
  Image,
  useToast,
} from '@chakra-ui/react';
import { SyntheticEvent, useCallback, useState } from 'react';
import { FileRejection, useDropzone } from 'react-dropzone';
import { FaTrash } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import {
  SDState,
  setInitialImagePath,
  setMaskPath,
} from '../../features/sd/sdSlice';
import MaskUploader from './MaskUploader';
import './InitImage.css';
import { uploadInitialImage } from '../../app/socketio';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

const sdSelector = createSelector(
  (state: RootState) => state.sd,
  (sd: SDState) => {
    return {
      initialImagePath: sd.initialImagePath,
      maskPath: sd.maskPath,
    };
  },
  { memoizeOptions: { resultEqualityCheck: isEqual } }
);

const InitImage = () => {
  const toast = useToast();
  const dispatch = useAppDispatch();
  const { initialImagePath, maskPath } = useAppSelector(sdSelector);

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: Array<FileRejection>) => {
      fileRejections.forEach((rejection: FileRejection) => {
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
      });

      acceptedFiles.forEach((file: File) => {
        dispatch(uploadInitialImage(file));
      });
    },
    [dispatch, toast]
  );

  const { getRootProps, getInputProps, open } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg', '.png'],
    },
  });

  const [shouldShowMask, setShouldShowMask] = useState<boolean>(false);
  const handleClickUploadIcon = (e: SyntheticEvent) => {
    e.stopPropagation();
    open();
  };
  const handleClickResetInitialImageAndMask = (e: SyntheticEvent) => {
    e.stopPropagation();
    dispatch(setInitialImagePath(''));
    dispatch(setMaskPath(''));
  };

  const handleMouseOverInitialImageUploadButton = () =>
    setShouldShowMask(false);
  const handleMouseOutInitialImageUploadButton = () => setShouldShowMask(true);

  const handleMouseOverMaskUploadButton = () => setShouldShowMask(true);
  const handleMouseOutMaskUploadButton = () => setShouldShowMask(true);

  return (
    <Flex
      {...getRootProps({
        onClick: initialImagePath ? (e) => e.stopPropagation() : undefined,
      })}
      direction={'column'}
      alignItems={'center'}
      gap={2}
    >
      <input {...getInputProps({ multiple: false })} />
      <Flex gap={2} justifyContent={'space-between'} width={'100%'}>
        <Button
          size={'sm'}
          fontSize={'md'}
          fontWeight={'normal'}
          onClick={handleClickUploadIcon}
          onMouseOver={handleMouseOverInitialImageUploadButton}
          onMouseOut={handleMouseOutInitialImageUploadButton}
        >
          Upload Image
        </Button>

        <MaskUploader>
          <Button
            size={'sm'}
            fontSize={'md'}
            fontWeight={'normal'}
            onClick={handleClickUploadIcon}
            onMouseOver={handleMouseOverMaskUploadButton}
            onMouseOut={handleMouseOutMaskUploadButton}
          >
            Upload Mask
          </Button>
        </MaskUploader>
        <IconButton
          size={'sm'}
          aria-label={'Reset initial image and mask'}
          onClick={handleClickResetInitialImageAndMask}
          icon={<FaTrash />}
        />
      </Flex>
      {initialImagePath && (
        <Flex position={'relative'} width={'100%'}>
          <Image
            fit={'contain'}
            src={initialImagePath}
            rounded={'md'}
            className={'checkerboard'}
          />
          {shouldShowMask && maskPath && (
            <Image
              position={'absolute'}
              top={0}
              left={0}
              fit={'contain'}
              src={maskPath}
              rounded={'md'}
              zIndex={1}
              className={'checkerboard'}
            />
          )}
        </Flex>
      )}
    </Flex>
  );
};

export default InitImage;
