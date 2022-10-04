import { Button } from '@chakra-ui/react';
import { MdPhotoLibrary } from 'react-icons/md';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch } from '../../app/store';
import { useAppSelector } from '../../app/store';
import HoverableImage from './HoverableImage';

/**
 * Simple image gallery.
 */
const ImageGallery = () => {
  const { images, currentImageUuid, areMoreImagesAvailable } = useAppSelector(
    (state: RootState) => state.gallery
  );
  const dispatch = useAppDispatch();
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

  return (
    <div className="image-gallery-container">
      {images.length ? (
        <>
          <p>
            <strong>Your Invocations</strong>
          </p>
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
        </>
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
  );
};

export default ImageGallery;
