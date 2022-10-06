import { Image } from '@chakra-ui/react';
import React from 'react';
import { RootState, useAppSelector } from '../../../app/store';

export default function InitialImageOverlay() {
  const initialImagePath = useAppSelector(
    (state: RootState) => state.options.initialImagePath
  );

  return initialImagePath ? (
    <Image
      fit={'contain'}
      src={initialImagePath}
      rounded={'md'}
      className={'checkerboard'}
    />
  ) : null;
}
