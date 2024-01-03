// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

import Accelerate
import Foundation

extension CVPixelBuffer {
  var size: CGSize {
    return CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
  }

  /// Returns a new `CVPixelBuffer` created by taking the self area and resizing it to the
  /// specified target size. Aspect ratios of source image and destination image are expected to be
  /// same.
  ///
  /// - Parameters:
  ///   - from: Source area of image to be cropped and resized.
  ///   - to: Size to scale the image to(i.e. image size used while training the model).
  /// - Returns: The cropped and resized image of itself.
  func resize(from source: CGRect, to size: CGSize) -> CVPixelBuffer? {
    let rect = CGRect(origin: CGPoint(x: 0, y: 0), size: self.size)
    guard rect.contains(source) else {
      os_log("Resizing Error: source area is out of index", type: .error)
      return nil
    }
    guard rect.size.width / rect.size.height - source.size.width / source.size.height < 1e-5
    else {
      os_log(
        "Resizing Error: source image ratio and destination image ratio is different",
        type: .error)
      return nil
    }

    let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
    let imageChannels = 4

    CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
    defer { CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0)) }

    // Finds the address of the upper leftmost pixel of the source area.
    guard
      let inputBaseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(
        by: Int(source.minY) * inputImageRowBytes + Int(source.minX) * imageChannels)
    else {
      return nil
    }

    // Crops given area as vImage Buffer.
    var croppedImage = vImage_Buffer(
      data: inputBaseAddress, height: UInt(source.height), width: UInt(source.width),
      rowBytes: inputImageRowBytes)

    let resultRowBytes = Int(size.width) * imageChannels
    guard let resultAddress = malloc(Int(size.height) * resultRowBytes) else {
      return nil
    }

    // Allocates a vacant vImage buffer for resized image.
    var resizedImage = vImage_Buffer(
      data: resultAddress,
      height: UInt(size.height), width: UInt(size.width),
      rowBytes: resultRowBytes
    )

    // Performs the scale operation on cropped image and stores it in result image buffer.
    guard vImageScale_ARGB8888(&croppedImage, &resizedImage, nil, vImage_Flags(0)) == kvImageNoError
    else {
      return nil
    }

    let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
      if let pointer = pointer {
        free(UnsafeMutableRawPointer(mutating: pointer))
      }
    }

    var result: CVPixelBuffer?

    // Converts the thumbnail vImage buffer to CVPixelBuffer
    let conversionStatus = CVPixelBufferCreateWithBytes(
      nil,
      Int(size.width), Int(size.height),
      CVPixelBufferGetPixelFormatType(self),
      resultAddress,
      resultRowBytes,
      releaseCallBack,
      nil,
      nil,
      &result
    )

    guard conversionStatus == kCVReturnSuccess else {
      free(resultAddress)
      return nil
    }

    return result
  }

  /// Returns the RGB `Data` representation of the given image buffer.
  ///
  /// - Parameters:
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  func rgbData(
    isModelQuantized: Bool
  ) -> Data? {
    CVPixelBufferLockBaseAddress(self, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
    guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
      return nil
    }

    let width = CVPixelBufferGetWidth(self)
    let height = CVPixelBufferGetHeight(self)
    let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
    let destinationBytesPerRow = Constants.rgbPixelChannels * width

    // Assign input image to `sourceBuffer` to convert it.
    var sourceBuffer = vImage_Buffer(
      data: sourceData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: sourceBytesPerRow)

    // Make `destinationBuffer` and `destinationData` for its data to be assigned.
    guard let destinationData = malloc(height * destinationBytesPerRow) else {
      os_log("Error: out of memory", type: .error)
      return nil
    }
    defer { free(destinationData) }
    var destinationBuffer = vImage_Buffer(
      data: destinationData,
      height: vImagePixelCount(height),
      width: vImagePixelCount(width),
      rowBytes: destinationBytesPerRow)

    // Convert image type.
    switch CVPixelBufferGetPixelFormatType(self) {
    case kCVPixelFormatType_32BGRA:
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    case kCVPixelFormatType_32ARGB:
      vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
    default:
      os_log("The type of this image is not supported.", type: .error)
      return nil
    }

    // Make `Data` with converted image.
    let imageByteData = Data(
      bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)

    if isModelQuantized { return imageByteData }

    let imageBytes = [UInt8](imageByteData)
    return Data(copyingBufferOf: imageBytes.map { Float($0) / Constants.maxRGBValue })
  }
}
