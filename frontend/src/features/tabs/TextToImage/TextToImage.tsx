import React from 'react';
import TextToImagePanel from './TextToImagePanel';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import ImageGallery from '../../gallery/ImageGallery';
import { RootState, useAppSelector } from '../../../app/store';

export default function TextToImage() {
  const shouldShowGallery = useAppSelector(
    (state: RootState) => state.options.shouldShowGallery
  );

  return (
    <div className="text-to-image-workarea">
      <TextToImagePanel />
      <div
        className="text-to-image-display"
        style={
          shouldShowGallery
            ? { gridTemplateColumns: 'auto max-content' }
            : { gridTemplateColumns: 'auto' }
        }
      >
        <CurrentImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
