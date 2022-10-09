import React from 'react';
import ImageToImagePanel from './ImageToImagePanel';
import ImageToImageDisplay from './ImageToImageDisplay';
import ImageGallery from '../../gallery/ImageGallery';
import { RootState, useAppSelector } from '../../../app/store';

export default function ImageToImage() {
  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  return (
    <div className="image-to-image-workarea">
      <ImageToImagePanel />
      <div
        className="image-to-image-display-area"
        style={
          shouldShowGallery
            ? { gridTemplateColumns: 'auto max-content' }
            : { gridTemplateColumns: 'auto' }
        }
      >
        <ImageToImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
