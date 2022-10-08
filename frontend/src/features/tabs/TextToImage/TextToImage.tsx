import React from 'react';
import TextToImagePanel from './TextToImagePanel';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import ImageGallery from '../../gallery/ImageGallery';

export default function TextToImage() {
  return (
    <div className="text-to-image-workarea">
      <TextToImagePanel />
      <div className="text-to-image-display">
        <CurrentImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
