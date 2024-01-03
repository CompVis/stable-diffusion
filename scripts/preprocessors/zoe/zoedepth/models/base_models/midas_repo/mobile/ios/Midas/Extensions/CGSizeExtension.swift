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

extension CGSize {
  /// Returns `CGAfineTransform` to resize `self` to fit in destination size, keeping aspect ratio
  /// of `self`. `self` image is resized to be inscribe to destination size and located in center of
  /// destination.
  ///
  /// - Parameter toFitIn: destination size to be filled.
  /// - Returns: `CGAffineTransform` to transform `self` image to `dest` image.
  func transformKeepAspect(toFitIn dest: CGSize) -> CGAffineTransform {
    let sourceRatio = self.height / self.width
    let destRatio = dest.height / dest.width

    // Calculates ratio `self` to `dest`.
    var ratio: CGFloat
    var x: CGFloat = 0
    var y: CGFloat = 0
    if sourceRatio > destRatio {
      // Source size is taller than destination. Resized to fit in destination height, and find
      // horizontal starting point to be centered.
      ratio = dest.height / self.height
      x = (dest.width - self.width * ratio) / 2
    } else {
      ratio = dest.width / self.width
      y = (dest.height - self.height * ratio) / 2
    }
    return CGAffineTransform(a: ratio, b: 0, c: 0, d: ratio, tx: x, ty: y)
  }
}
