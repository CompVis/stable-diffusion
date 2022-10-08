import React from 'react';
import ImageGallery from '../../gallery/ImageGallery';
import ImageToImageDisplay from './ImageToImageDisplay';

import ImageToImagePanel from './ImageToImagePanel';

export default function ImageToImage() {
  return (
    <div className="image-to-image-workarea">
      <ImageToImagePanel />
      <ImageToImageDisplay />
      <ImageGallery />
    </div>
  );
}
