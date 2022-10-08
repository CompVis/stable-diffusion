import {
  Button,
  Drawer,
  DrawerBody,
  DrawerCloseButton,
  DrawerContent,
  DrawerHeader,
  useDisclosure,
} from '@chakra-ui/react';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdPhotoLibrary } from 'react-icons/md';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch } from '../../app/store';
import { useAppSelector } from '../../app/store';
import { selectNextImage, selectPrevImage } from './gallerySlice';
import HoverableImage from './HoverableImage';

/**
 * Simple image gallery.
 */
const ImageGalleryOld = () => {
  const { images, currentImageUuid, areMoreImagesAvailable } = useAppSelector(
    (state: RootState) => state.gallery
  );
  const dispatch = useAppDispatch();

  const { isOpen, onOpen, onClose } = useDisclosure();

  /**
   * I don't like that this needs to rerender whenever the current image is changed.
   * What if we have a large number of images? I suppose pagination (planned) will
   * mitigate this issue.
   *
   * TODO: Refactor if performance complaints, or after migrating to new API which supports pagination.
   */

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  useHotkeys(
    'g',
    () => {
      if (isOpen) {
        onClose();
      } else {
        onOpen();
      }
    },
    [isOpen]
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
      <Button
        colorScheme="teal"
        onClick={onOpen}
        className="image-gallery-popup-btn"
      >
        <MdPhotoLibrary />
      </Button>
      <Drawer
        isOpen={isOpen}
        placement="right"
        onClose={onClose}
        autoFocus={false}
        trapFocus={false}
        closeOnOverlayClick={false}
      >
        <DrawerContent className="image-gallery-popup">
          <div className="image-gallery-header">
            <DrawerHeader>Your Invocations</DrawerHeader>
            <DrawerCloseButton />
          </div>
          <DrawerBody className="image-gallery-body">
            <div className="image-gallery-container">
              {images.length ? (
                <div className="image-gallery">
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
                </div>
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
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </div>
  );
};

export default ImageGalleryOld;
