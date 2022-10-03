import React from 'react';
import Img2ImgPlaceHolder from '../../../assets/images/image2img.png';

export const ImageToImageWIP = () => {
  return (
    <div className="work-in-progress txt2img-work-in-progress">
      <img src={Img2ImgPlaceHolder} alt="img2img_placeholder" />
      <h1>Image To Image</h1>
      <p>
        Image to Image is already available in the WebUI. You can access it from
        the Text to Image - Advanced Options menu. A dedicated UI for Image To
        Image will be released soon.
      </p>
    </div>
  );
};
