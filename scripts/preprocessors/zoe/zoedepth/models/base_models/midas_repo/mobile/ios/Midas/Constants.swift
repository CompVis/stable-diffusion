// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

enum Constants {
  // MARK: - Constants related to the image processing
  static let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
  static let rgbPixelChannels = 3
  static let maxRGBValue: Float32 = 255.0

  // MARK: - Constants related to the model interperter
  static let defaultThreadCount = 2
  static let defaultDelegate: Delegates = .CPU
}
