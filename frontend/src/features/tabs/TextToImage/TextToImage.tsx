import React from 'react';
import CurrentImageDisplay from '../../gallery/CurrentImageDisplay';
import ImageGallery from '../../gallery/ImageGallery';
import TextToImagePanel from './TextToImagePanel';

export default function TextToImage() {
  return (
    <div className="text-to-image-workarea">
      <TextToImagePanel />
      <CurrentImageDisplay />
      <ImageGallery />
    </div>
  );
}
