import { Image } from '@chakra-ui/react';
import { useAppSelector } from '../../app/store';
import { RootState } from '../../app/store';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const [shouldShowImageDetails, setShouldShowImageDetails] =
    useState<boolean>(false);

  const imageToDisplay = intermediateImage || currentImage;

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
      </div>
    </div>
  ) : (
    <div className="current-image-display-placeholder">
      <MdPhoto />
    </div>
  );
};

export default CurrentImageDisplay;
