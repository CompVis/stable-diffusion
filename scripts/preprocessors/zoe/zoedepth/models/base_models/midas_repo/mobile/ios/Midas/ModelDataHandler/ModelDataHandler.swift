// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Accelerate
import CoreImage
import Foundation
import TensorFlowLite
import UIKit

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained.
class ModelDataHandler {
  // MARK: - Private Properties

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// TensorFlow lite `Tensor` of model input and output.
  private var inputTensor: Tensor

  //private var heatsTensor: Tensor
  //private var offsetsTensor: Tensor
  private var outputTensor: Tensor
  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model is
  /// successfully loaded from the app's main bundle. Default `threadCount` is 2.
  init(
    threadCount: Int = Constants.defaultThreadCount,
    delegate: Delegates = Constants.defaultDelegate
  ) throws {
    // Construct the path to the model file.
    guard
      let modelPath = Bundle.main.path(
        forResource: Model.file.name,
        ofType: Model.file.extension
      )
    else {
      fatalError("Failed to load the model file with name: \(Model.file.name).")
    }

    // Specify the options for the `Interpreter`.
    var options = Interpreter.Options()
    options.threadCount = threadCount

    // Specify the delegates for the `Interpreter`.
    var delegates: [Delegate]?
    switch delegate {
    case .Metal:
      delegates = [MetalDelegate()]
    case .CoreML:
      if let coreMLDelegate = CoreMLDelegate() {
        delegates = [coreMLDelegate]
      } else {
        delegates = nil
      }
    default:
      delegates = nil
    }

    // Create the `Interpreter`.
    interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: delegates)

    // Initialize input and output `Tensor`s.
    // Allocate memory for the model's input `Tensor`s.
    try interpreter.allocateTensors()

    // Get allocated input and output `Tensor`s.
    inputTensor = try interpreter.input(at: 0)
    outputTensor = try interpreter.output(at: 0)
    //heatsTensor = try interpreter.output(at: 0)
    //offsetsTensor = try interpreter.output(at: 1)

    /*
    // Check if input and output `Tensor`s are in the expected formats.
    guard (inputTensor.dataType == .uInt8) == Model.isQuantized else {
      fatalError("Unexpected Model: quantization is \(!Model.isQuantized)")
    }

    guard inputTensor.shape.dimensions[0] == Model.input.batchSize,
      inputTensor.shape.dimensions[1] == Model.input.height,
      inputTensor.shape.dimensions[2] == Model.input.width,
      inputTensor.shape.dimensions[3] == Model.input.channelSize
    else {
      fatalError("Unexpected Model: input shape")
    }

    
    guard heatsTensor.shape.dimensions[0] == Model.output.batchSize,
      heatsTensor.shape.dimensions[1] == Model.output.height,
      heatsTensor.shape.dimensions[2] == Model.output.width,
      heatsTensor.shape.dimensions[3] == Model.output.keypointSize
    else {
      fatalError("Unexpected Model: heat tensor")
    }

    guard offsetsTensor.shape.dimensions[0] == Model.output.batchSize,
      offsetsTensor.shape.dimensions[1] == Model.output.height,
      offsetsTensor.shape.dimensions[2] == Model.output.width,
      offsetsTensor.shape.dimensions[3] == Model.output.offsetSize
    else {
      fatalError("Unexpected Model: offset tensor")
    }
 */

  }

  /// Runs Midas model with given image with given source area to destination area.
  ///
  /// - Parameters:
  ///   - on: Input image to run the model.
  ///   - from: Range of input image to run the model.
  ///   - to: Size of view to render the result.
  /// - Returns: Result of the inference and the times consumed in every steps.
  func runMidas(on pixelbuffer: CVPixelBuffer, from source: CGRect, to dest: CGSize)
    //-> (Result, Times)?
    //-> (FlatArray<Float32>, Times)?
    -> ([Float], Int, Int, Times)?
  {
    // Start times of each process.
    let preprocessingStartTime: Date
    let inferenceStartTime: Date
    let postprocessingStartTime: Date

    // Processing times in miliseconds.
    let preprocessingTime: TimeInterval
    let inferenceTime: TimeInterval
    let postprocessingTime: TimeInterval

    preprocessingStartTime = Date()
    guard let data = preprocess(of: pixelbuffer, from: source) else {
      os_log("Preprocessing failed", type: .error)
      return nil
    }
    preprocessingTime = Date().timeIntervalSince(preprocessingStartTime) * 1000

    inferenceStartTime = Date()
    inference(from: data)
    inferenceTime = Date().timeIntervalSince(inferenceStartTime) * 1000

    postprocessingStartTime = Date()
    //guard let result = postprocess(to: dest) else {
    //  os_log("Postprocessing failed", type: .error)
    //  return nil
    //}
    postprocessingTime = Date().timeIntervalSince(postprocessingStartTime) * 1000


    let results: [Float]
    switch outputTensor.dataType {
    case .uInt8:
      guard let quantization = outputTensor.quantizationParameters else {
        print("No results returned because the quantization values for the output tensor are nil.")
        return nil
      }
      let quantizedResults = [UInt8](outputTensor.data)
      results = quantizedResults.map {
        quantization.scale * Float(Int($0) - quantization.zeroPoint)
      }
    case .float32:
      results = [Float32](unsafeData: outputTensor.data) ?? []
    default:
      print("Output tensor data type \(outputTensor.dataType) is unsupported for this example app.")
      return nil
    }
    
    
    let times = Times(
      preprocessing: preprocessingTime,
      inference: inferenceTime,
      postprocessing: postprocessingTime)

    return (results, Model.input.width, Model.input.height, times)
  }

  // MARK: - Private functions to run model
  /// Preprocesses given rectangle image to be `Data` of disired size by croping and resizing it.
  ///
  /// - Parameters:
  ///   - of: Input image to crop and resize.
  ///   - from: Target area to be cropped and resized.
  /// - Returns: The cropped and resized image. `nil` if it can not be processed.
  private func preprocess(of pixelBuffer: CVPixelBuffer, from targetSquare: CGRect) -> Data? {
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32BGRA)

    // Resize `targetSquare` of input image to `modelSize`.
    let modelSize = CGSize(width: Model.input.width, height: Model.input.height)
    guard let thumbnail = pixelBuffer.resize(from: targetSquare, to: modelSize)
    else {
      return nil
    }

    // Remove the alpha component from the image buffer to get the initialized `Data`.
    let byteCount =
      Model.input.batchSize
      * Model.input.height * Model.input.width
      * Model.input.channelSize
    guard
      let inputData = thumbnail.rgbData(
        isModelQuantized: Model.isQuantized
      )
    else {
      os_log("Failed to convert the image buffer to RGB data.", type: .error)
      return nil
    }

    return inputData
  }

   
    
    /*
  /// Postprocesses output `Tensor`s to `Result` with size of view to render the result.
  ///
  /// - Parameters:
  ///   - to: Size of view to be displaied.
  /// - Returns: Postprocessed `Result`. `nil` if it can not be processed.
  private func postprocess(to viewSize: CGSize) -> Result? {
    // MARK: Formats output tensors
    // Convert `Tensor` to `FlatArray`. As Midas is not quantized, convert them to Float type
    // `FlatArray`.
    let heats = FlatArray<Float32>(tensor: heatsTensor)
    let offsets = FlatArray<Float32>(tensor: offsetsTensor)

    // MARK: Find position of each key point
    // Finds the (row, col) locations of where the keypoints are most likely to be. The highest
    // `heats[0, row, col, keypoint]` value, the more likely `keypoint` being located in (`row`,
    // `col`).
    let keypointPositions = (0..<Model.output.keypointSize).map { keypoint -> (Int, Int) in
      var maxValue = heats[0, 0, 0, keypoint]
      var maxRow = 0
      var maxCol = 0
      for row in 0..<Model.output.height {
        for col in 0..<Model.output.width {
          if heats[0, row, col, keypoint] > maxValue {
            maxValue = heats[0, row, col, keypoint]
            maxRow = row
            maxCol = col
          }
        }
      }
      return (maxRow, maxCol)
    }

    // MARK: Calculates total confidence score
    // Calculates total confidence score of each key position.
    let totalScoreSum = keypointPositions.enumerated().reduce(0.0) { accumulator, elem -> Float32 in
      accumulator + sigmoid(heats[0, elem.element.0, elem.element.1, elem.offset])
    }
    let totalScore = totalScoreSum / Float32(Model.output.keypointSize)

    // MARK: Calculate key point position on model input
    // Calculates `KeyPoint` coordination model input image with `offsets` adjustment.
    let coords = keypointPositions.enumerated().map { index, elem -> (y: Float32, x: Float32) in
      let (y, x) = elem
      let yCoord =
        Float32(y) / Float32(Model.output.height - 1) * Float32(Model.input.height)
        + offsets[0, y, x, index]
      let xCoord =
        Float32(x) / Float32(Model.output.width - 1) * Float32(Model.input.width)
        + offsets[0, y, x, index + Model.output.keypointSize]
      return (y: yCoord, x: xCoord)
    }

    // MARK: Transform key point position and make lines
    // Make `Result` from `keypointPosition'. Each point is adjusted to `ViewSize` to be drawn.
    var result = Result(dots: [], lines: [], score: totalScore)
    var bodyPartToDotMap = [BodyPart: CGPoint]()
    for (index, part) in BodyPart.allCases.enumerated() {
      let position = CGPoint(
        x: CGFloat(coords[index].x) * viewSize.width / CGFloat(Model.input.width),
        y: CGFloat(coords[index].y) * viewSize.height / CGFloat(Model.input.height)
      )
      bodyPartToDotMap[part] = position
      result.dots.append(position)
    }

    do {
      try result.lines = BodyPart.lines.map { map throws -> Line in
        guard let from = bodyPartToDotMap[map.from] else {
          throw PostprocessError.missingBodyPart(of: map.from)
        }
        guard let to = bodyPartToDotMap[map.to] else {
          throw PostprocessError.missingBodyPart(of: map.to)
        }
        return Line(from: from, to: to)
      }
    } catch PostprocessError.missingBodyPart(let missingPart) {
      os_log("Postprocessing error: %s is missing.", type: .error, missingPart.rawValue)
      return nil
    } catch {
      os_log("Postprocessing error: %s", type: .error, error.localizedDescription)
      return nil
    }

    return result
  }
*/
    

    
  /// Run inference with given `Data`
  ///
  /// Parameter `from`: `Data` of input image to run model.
  private func inference(from data: Data) {
    // Copy the initialized `Data` to the input `Tensor`.
    do {
      try interpreter.copy(data, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
      //heatsTensor = try interpreter.output(at: 0)
      //offsetsTensor = try interpreter.output(at: 1)
        

    } catch let error {
      os_log(
        "Failed to invoke the interpreter with error: %s", type: .error,
        error.localizedDescription)
      return
    }
  }

  /// Returns value within [0,1].
  private func sigmoid(_ x: Float32) -> Float32 {
    return (1.0 / (1.0 + exp(-x)))
  }
}

// MARK: - Data types for inference result
struct KeyPoint {
  var bodyPart: BodyPart = BodyPart.NOSE
  var position: CGPoint = CGPoint()
  var score: Float = 0.0
}

struct Line {
  let from: CGPoint
  let to: CGPoint
}

struct Times {
  var preprocessing: Double
  var inference: Double
  var postprocessing: Double
}

struct Result {
  var dots: [CGPoint]
  var lines: [Line]
  var score: Float
}

enum BodyPart: String, CaseIterable {
  case NOSE = "nose"
  case LEFT_EYE = "left eye"
  case RIGHT_EYE = "right eye"
  case LEFT_EAR = "left ear"
  case RIGHT_EAR = "right ear"
  case LEFT_SHOULDER = "left shoulder"
  case RIGHT_SHOULDER = "right shoulder"
  case LEFT_ELBOW = "left elbow"
  case RIGHT_ELBOW = "right elbow"
  case LEFT_WRIST = "left wrist"
  case RIGHT_WRIST = "right wrist"
  case LEFT_HIP = "left hip"
  case RIGHT_HIP = "right hip"
  case LEFT_KNEE = "left knee"
  case RIGHT_KNEE = "right knee"
  case LEFT_ANKLE = "left ankle"
  case RIGHT_ANKLE = "right ankle"

  /// List of lines connecting each part.
  static let lines = [
    (from: BodyPart.LEFT_WRIST, to: BodyPart.LEFT_ELBOW),
    (from: BodyPart.LEFT_ELBOW, to: BodyPart.LEFT_SHOULDER),
    (from: BodyPart.LEFT_SHOULDER, to: BodyPart.RIGHT_SHOULDER),
    (from: BodyPart.RIGHT_SHOULDER, to: BodyPart.RIGHT_ELBOW),
    (from: BodyPart.RIGHT_ELBOW, to: BodyPart.RIGHT_WRIST),
    (from: BodyPart.LEFT_SHOULDER, to: BodyPart.LEFT_HIP),
    (from: BodyPart.LEFT_HIP, to: BodyPart.RIGHT_HIP),
    (from: BodyPart.RIGHT_HIP, to: BodyPart.RIGHT_SHOULDER),
    (from: BodyPart.LEFT_HIP, to: BodyPart.LEFT_KNEE),
    (from: BodyPart.LEFT_KNEE, to: BodyPart.LEFT_ANKLE),
    (from: BodyPart.RIGHT_HIP, to: BodyPart.RIGHT_KNEE),
    (from: BodyPart.RIGHT_KNEE, to: BodyPart.RIGHT_ANKLE),
  ]
}

// MARK: - Delegates Enum
enum Delegates: Int, CaseIterable {
  case CPU
  case Metal
  case CoreML

  var description: String {
    switch self {
    case .CPU:
      return "CPU"
    case .Metal:
      return "GPU"
    case .CoreML:
      return "NPU"
    }
  }
}

// MARK: - Custom Errors
enum PostprocessError: Error {
  case missingBodyPart(of: BodyPart)
}

// MARK: - Information about the model file.
typealias FileInfo = (name: String, extension: String)

enum Model {
  static let file: FileInfo = (
    name: "model_opt", extension: "tflite"
  )

  static let input = (batchSize: 1, height: 256, width: 256, channelSize: 3)
  static let output = (batchSize: 1, height: 256, width: 256, channelSize: 1)
  static let isQuantized = false
}


extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {
    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
