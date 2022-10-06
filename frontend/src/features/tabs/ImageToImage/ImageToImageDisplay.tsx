import React from 'react';
import { FaUpload } from 'react-icons/fa';
import { uploadInitialImage } from '../../../app/socketio/actions';
import { RootState, useAppSelector } from '../../../app/store';
import InvokeImageUploader from '../../../common/components/InvokeImageUploader';
import CurrentImageButtons from '../../gallery/CurrentImageButtons';
import CurrentImagePreview from '../../gallery/CurrentImagePreview';
import ImageMetadataViewer from '../../gallery/ImageMetaDataViewer/ImageMetadataViewer';

import InitImagePreview from './InitImagePreview';

export default function ImageToImageDisplay() {
  const initialImagePath = useAppSelector(
    (state: RootState) => state.options.initialImagePath
  );

  const { currentImage, intermediateImage } = useAppSelector(
    (state: RootState) => state.gallery
  );

  const shouldShowImageDetails = useAppSelector(
    (state: RootState) => state.options.shouldShowImageDetails
  );

  const imageToDisplay = intermediateImage || currentImage;

  return (
    <div
      className="image-to-image-display"
      style={
        imageToDisplay
          ? { gridAutoRows: 'max-content auto' }
          : { gridAutoRows: 'auto' }
      }
    >
      {initialImagePath ? (
        <>
          {imageToDisplay ? (
            <>
              <CurrentImageButtons image={imageToDisplay} />
              <div className="image-to-image-dual-preview-container">
                <div className="image-to-image-dual-preview">
                  <InitImagePreview />
                  <div className="image-to-image-current-image-display">
                    <CurrentImagePreview imageToDisplay={imageToDisplay} />
                  </div>
                </div>
                {shouldShowImageDetails && (
                  <ImageMetadataViewer
                    image={imageToDisplay}
                    styleClass="img2img-metadata"
                  />
                )}
              </div>
            </>
          ) : (
            <div className="image-to-image-single-preview">
              <InitImagePreview />
            </div>
          )}
        </>
      ) : (
        <div className="upload-image">
          <InvokeImageUploader
            label="Upload or Drop Image Here"
            icon={<FaUpload />}
            styleClass="image-to-image-upload-btn"
            dispatcher={uploadInitialImage}
          />
        </div>
      )}
    </div>
  );
}
