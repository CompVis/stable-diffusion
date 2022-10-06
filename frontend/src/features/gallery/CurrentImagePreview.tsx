import { IconButton, Image } from '@chakra-ui/react';
import React, { useState } from 'react';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import { GalleryState, selectNextImage, selectPrevImage } from './gallerySlice';
import * as InvokeAI from '../../app/invokeai';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';

const imagesSelector = createSelector(
  (state: RootState) => state.gallery,
  (gallery: GalleryState) => {
    const currentImageIndex = gallery.images.findIndex(
      (i) => i.uuid === gallery?.currentImage?.uuid
    );
    const imagesLength = gallery.images.length;
    return {
      isOnFirstImage: currentImageIndex === 0,
      isOnLastImage:
        !isNaN(currentImageIndex) && currentImageIndex === imagesLength - 1,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

interface CurrentImagePreviewProps {
  imageToDisplay: InvokeAI.Image;
}

export default function CurrentImagePreview(props: CurrentImagePreviewProps) {
  const { imageToDisplay } = props;
  const dispatch = useAppDispatch();

  const { isOnFirstImage, isOnLastImage } = useAppSelector(imagesSelector);

  const shouldShowImageDetails = useAppSelector(
    (state: RootState) => state.options.shouldShowImageDetails
  );

  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const handleCurrentImagePreviewMouseOver = () => {
    setShouldShowNextPrevButtons(true);
  };

  const handleCurrentImagePreviewMouseOut = () => {
    setShouldShowNextPrevButtons(false);
  };

  const handleClickPrevButton = () => {
    dispatch(selectPrevImage());
  };

  const handleClickNextButton = () => {
    dispatch(selectNextImage());
  };

  return (
    <div className="current-image-preview">
      <Image
        src={imageToDisplay.url}
        fit="contain"
        maxWidth={'100%'}
        maxHeight={'100%'}
      />
      {!shouldShowImageDetails && (
        <div className="current-image-next-prev-buttons">
          <div
            className="next-prev-button-trigger-area prev-button-trigger-area"
            onMouseOver={handleCurrentImagePreviewMouseOver}
            onMouseOut={handleCurrentImagePreviewMouseOut}
          >
            {shouldShowNextPrevButtons && !isOnFirstImage && (
              <IconButton
                aria-label="Previous image"
                icon={<FaAngleLeft className="next-prev-button" />}
                variant="unstyled"
                onClick={handleClickPrevButton}
              />
            )}
          </div>
          <div
            className="next-prev-button-trigger-area next-button-trigger-area"
            onMouseOver={handleCurrentImagePreviewMouseOver}
            onMouseOut={handleCurrentImagePreviewMouseOut}
          >
            {shouldShowNextPrevButtons && !isOnLastImage && (
              <IconButton
                aria-label="Next image"
                icon={<FaAngleRight className="next-prev-button" />}
                variant="unstyled"
                onClick={handleClickNextButton}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
