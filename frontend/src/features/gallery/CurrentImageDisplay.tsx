import { RootState, useAppSelector } from '../../app/store';
import CurrentImageButtons from './CurrentImageButtons';
import { MdPhoto } from 'react-icons/md';
import CurrentImagePreview from './CurrentImagePreview';
import ImageMetadataViewer from './ImageMetaDataViewer/ImageMetadataViewer';

/**
 * Displays the current image if there is one, plus associated actions.
 */
const CurrentImageDisplay = () => {
  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const shouldShowImageDetails = useAppSelector(
    (state: RootState) => state.options.shouldShowImageDetails
  );

  const imageToDisplay = intermediateImage || currentImage;

  return imageToDisplay ? (
    <div className="current-image-display">
      <div className="current-image-tools">
        <CurrentImageButtons image={imageToDisplay} />
      </div>
      <CurrentImagePreview imageToDisplay={imageToDisplay} />
      {shouldShowImageDetails && (
        <ImageMetadataViewer
          image={imageToDisplay}
          styleClass="current-image-metadata"
        />
      )}
    </div>
  ) : (
    <div className="current-image-display-placeholder">
      <MdPhoto />
    </div>
  );
};

export default CurrentImageDisplay;
