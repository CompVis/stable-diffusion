import { Button, IconButton } from '@chakra-ui/button';
import { Resizable } from 're-resizable';

import React, { useState } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdClear, MdPhotoLibrary } from 'react-icons/md';
import Masonry from 'react-masonry-css';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import IAIIconButton from '../../common/components/IAIIconButton';
import { selectNextImage, selectPrevImage } from './gallerySlice';
import HoverableImage from './HoverableImage';
import { setShouldShowGallery } from '../options/optionsSlice';

export default function ImageGallery() {
  const { images, currentImageUuid, areMoreImagesAvailable } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  const activeTab = useAppSelector(
    (state: RootState) => state.options.activeTab
  );

  const dispatch = useAppDispatch();

  const [column, setColumn] = useState<number | undefined>();

  const handleResize = (event: MouseEvent | TouchEvent | any) => {
    setColumn(Math.floor((window.innerWidth - event.x) / 120));
  };

  const handleShowGalleryToggle = () => {
    dispatch(setShouldShowGallery(!shouldShowGallery));
  };

  const handleGalleryClose = () => {
    dispatch(setShouldShowGallery(false));
  };

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  useHotkeys(
    'g',
    () => {
      handleShowGalleryToggle();
    },
    [shouldShowGallery]
  );

  useHotkeys(
    'left',
    () => {
      dispatch(selectPrevImage());
    },
    []
  );

  useHotkeys(
    'right',
    () => {
      dispatch(selectNextImage());
    },
    []
  );

  return (
    <div className="image-gallery-area">
      {!shouldShowGallery && (
        <IAIIconButton
          tooltip="Show Gallery"
          tooltipPlacement="top"
          aria-label="Show Gallery"
          onClick={handleShowGalleryToggle}
          className="image-gallery-popup-btn"
        >
          <MdPhotoLibrary />
        </IAIIconButton>
      )}

      {shouldShowGallery && (
        <Resizable
          defaultSize={{ width: '300', height: '100%' }}
          minWidth={'300'}
          maxWidth={activeTab == 1 ? '300' : '600'}
          className="image-gallery-popup"
          onResize={handleResize}
        >
          {/* <div className="image-gallery-popup"></div> */}
          <div className="image-gallery-header">
            <h1>Your Invocations</h1>
            <IconButton
              size={'sm'}
              aria-label={'Close Gallery'}
              onClick={handleGalleryClose}
              className="image-gallery-close-btn"
              icon={<MdClear />}
            />
          </div>
          <div className="image-gallery-container">
            {images.length ? (
              <Masonry
                className="masonry-grid"
                columnClassName="masonry-grid_column"
                breakpointCols={column}
              >
                {/* <div className="image-gallery"> */}
                {images.map((image) => {
                  const { uuid } = image;
                  const isSelected = currentImageUuid === uuid;
                  return (
                    <HoverableImage
                      key={uuid}
                      image={image}
                      isSelected={isSelected}
                    />
                  );
                })}
                {/* </div> */}
              </Masonry>
            ) : (
              <div className="image-gallery-container-placeholder">
                <MdPhotoLibrary />
                <p>No Images In Gallery</p>
              </div>
            )}
            <Button
              onClick={handleClickLoadMore}
              isDisabled={!areMoreImagesAvailable}
              className="image-gallery-load-more-btn"
            >
              {areMoreImagesAvailable ? 'Load More' : 'All Images Loaded'}
            </Button>
          </div>
        </Resizable>
      )}
    </div>
  );
}
