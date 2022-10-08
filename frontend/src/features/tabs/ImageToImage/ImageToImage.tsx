import React from 'react';
import ImageToImagePanel from './ImageToImagePanel';
import ImageToImageDisplay from './ImageToImageDisplay';
import ImageGallery from '../../gallery/ImageGallery';

export default function ImageToImage() {
  return (
    <div className="image-to-image-workarea">
      <ImageToImagePanel />
      <div className="image-to-image-display-area">
        <ImageToImageDisplay />
        <ImageGallery />
      </div>
    </div>
  );
}
