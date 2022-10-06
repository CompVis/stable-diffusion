import { IconButton, Image } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';
import { FaAngleLeft, FaAngleRight } from 'react-icons/fa';
import { selectNextImage, selectPrevImage } from './gallerySlice';

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const dispatch = useAppDispatch();
  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] =
    useState<boolean>(false);

  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const [shouldShowImageDetails, setShouldShowImageDetails] =
    useState<boolean>(false);

  const imageToDisplay = intermediateImage || currentImage;

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

  return imageToDisplay ? (
    <div className="current-image-display">
      <div className="current-image-tools">
        <CurrentImageButtons
          image={imageToDisplay}
          shouldShowImageDetails={shouldShowImageDetails}
          setShouldShowImageDetails={setShouldShowImageDetails}
        />
      </div>
      <div className="current-image-preview">
        <Image
          src={imageToDisplay.url}
          fit="contain"
          maxWidth={'100%'}
          maxHeight={'100%'}
        />
        {shouldShowImageDetails && (
          <div className="current-image-metadata-viewer">
            <ImageMetadataViewer image={imageToDisplay} />
          </div>
        )}
        {!shouldShowImageDetails && (
          <div className="current-image-next-prev-buttons">
            <div
              className="next-prev-button-trigger-area prev-button-trigger-area"
              onMouseOver={handleCurrentImagePreviewMouseOver}
              onMouseOut={handleCurrentImagePreviewMouseOut}
            >
              {shouldShowNextPrevButtons && (
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
              {shouldShowNextPrevButtons && (
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
    </div>
  ) : (
    <div className="current-image-display-placeholder">
      <MdPhoto />
    </div>
  );
};

export default CurrentImageDisplay;
