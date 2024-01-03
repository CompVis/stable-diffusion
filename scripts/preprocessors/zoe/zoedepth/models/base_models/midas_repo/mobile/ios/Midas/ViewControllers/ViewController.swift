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

import AVFoundation
import UIKit
import os


public struct PixelData {
    var a: UInt8
    var r: UInt8
    var g: UInt8
    var b: UInt8
}

extension UIImage {
    convenience init?(pixels: [PixelData], width: Int, height: Int) {
        guard width > 0 && height > 0, pixels.count == width * height else { return nil }
        var data = pixels
        guard let providerRef = CGDataProvider(data: Data(bytes: &data, count: data.count * MemoryLayout<PixelData>.size) as CFData)
            else { return nil }
        guard let cgim = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * MemoryLayout<PixelData>.size,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue),
            provider: providerRef,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent)
        else { return nil }
        self.init(cgImage: cgim)
    }
}


class ViewController: UIViewController {
  // MARK: Storyboards Connections
  @IBOutlet weak var previewView: PreviewView!

  //@IBOutlet weak var overlayView: OverlayView!
  @IBOutlet weak var overlayView: UIImageView!
        
  private var imageView : UIImageView = UIImageView(frame:CGRect(x:0, y:0, width:400, height:400))
    
  private var imageViewInitialized: Bool = false
    
  @IBOutlet weak var resumeButton: UIButton!
  @IBOutlet weak var cameraUnavailableLabel: UILabel!

  @IBOutlet weak var tableView: UITableView!

  @IBOutlet weak var threadCountLabel: UILabel!
  @IBOutlet weak var threadCountStepper: UIStepper!

  @IBOutlet weak var delegatesControl: UISegmentedControl!

  // MARK: ModelDataHandler traits
  var threadCount: Int = Constants.defaultThreadCount
  var delegate: Delegates = Constants.defaultDelegate

  // MARK: Result Variables
  // Inferenced data to render.
  private var inferencedData: InferencedData?

  // Minimum score to render the result.
  private let minimumScore: Float = 0.5
    
  private var avg_latency: Double = 0.0

  // Relative location of `overlayView` to `previewView`.
  private var overlayViewFrame: CGRect?

  private var previewViewFrame: CGRect?

  // MARK: Controllers that manage functionality
  // Handles all the camera related functionality
  private lazy var cameraCapture = CameraFeedManager(previewView: previewView)

  // Handles all data preprocessing and makes calls to run inference.
  private var modelDataHandler: ModelDataHandler?

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()

    do {
      modelDataHandler = try ModelDataHandler()
    } catch let error {
      fatalError(error.localizedDescription)
    }

    cameraCapture.delegate = self
    tableView.delegate = self
    tableView.dataSource = self

    // MARK: UI Initialization
    // Setup thread count stepper with white color.
    // https://forums.developer.apple.com/thread/121495
    threadCountStepper.setDecrementImage(
      threadCountStepper.decrementImage(for: .normal), for: .normal)
    threadCountStepper.setIncrementImage(
      threadCountStepper.incrementImage(for: .normal), for: .normal)
    // Setup initial stepper value and its label.
    threadCountStepper.value = Double(Constants.defaultThreadCount)
    threadCountLabel.text = Constants.defaultThreadCount.description

    // Setup segmented controller's color.
    delegatesControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.lightGray],
      for: .normal)
    delegatesControl.setTitleTextAttributes(
      [NSAttributedString.Key.foregroundColor: UIColor.black],
      for: .selected)
    // Remove existing segments to initialize it with `Delegates` entries.
    delegatesControl.removeAllSegments()
    Delegates.allCases.forEach { delegate in
      delegatesControl.insertSegment(
        withTitle: delegate.description,
        at: delegate.rawValue,
        animated: false)
    }
    delegatesControl.selectedSegmentIndex = 0
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)

    cameraCapture.checkCameraConfigurationAndStartSession()
  }

  override func viewWillDisappear(_ animated: Bool) {
    cameraCapture.stopSession()
  }

  override func viewDidLayoutSubviews() {
    overlayViewFrame = overlayView.frame
    previewViewFrame = previewView.frame
  }

  // MARK: Button Actions
  @IBAction func didChangeThreadCount(_ sender: UIStepper) {
    let changedCount = Int(sender.value)
    if threadCountLabel.text == changedCount.description {
      return
    }

    do {
      modelDataHandler = try ModelDataHandler(threadCount: changedCount, delegate: delegate)
    } catch let error {
      fatalError(error.localizedDescription)
    }
    threadCount = changedCount
    threadCountLabel.text = changedCount.description
    os_log("Thread count is changed to: %d", threadCount)
  }

  @IBAction func didChangeDelegate(_ sender: UISegmentedControl) {
    guard let changedDelegate = Delegates(rawValue: delegatesControl.selectedSegmentIndex) else {
      fatalError("Unexpected value from delegates segemented controller.")
    }
    do {
      modelDataHandler = try ModelDataHandler(threadCount: threadCount, delegate: changedDelegate)
    } catch let error {
      fatalError(error.localizedDescription)
    }
    delegate = changedDelegate
    os_log("Delegate is changed to: %s", delegate.description)
  }

  @IBAction func didTapResumeButton(_ sender: Any) {
    cameraCapture.resumeInterruptedSession { complete in

      if complete {
        self.resumeButton.isHidden = true
        self.cameraUnavailableLabel.isHidden = true
      } else {
        self.presentUnableToResumeSessionAlert()
      }
    }
  }

  func presentUnableToResumeSessionAlert() {
    let alert = UIAlertController(
      title: "Unable to Resume Session",
      message: "There was an error while attempting to resume session.",
      preferredStyle: .alert
    )
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    self.present(alert, animated: true)
  }
}

// MARK: - CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {
  func cameraFeedManager(_ manager: CameraFeedManager, didOutput pixelBuffer: CVPixelBuffer) {
    runModel(on: pixelBuffer)
  }

  // MARK: Session Handling Alerts
  func cameraFeedManagerDidEncounterSessionRunTimeError(_ manager: CameraFeedManager) {
    // Handles session run time error by updating the UI and providing a button if session can be
    // manually resumed.
    self.resumeButton.isHidden = false
  }

  func cameraFeedManager(
    _ manager: CameraFeedManager, sessionWasInterrupted canResumeManually: Bool
  ) {
    // Updates the UI when session is interupted.
    if canResumeManually {
      self.resumeButton.isHidden = false
    } else {
      self.cameraUnavailableLabel.isHidden = false
    }
  }

  func cameraFeedManagerDidEndSessionInterruption(_ manager: CameraFeedManager) {
    // Updates UI once session interruption has ended.
    self.cameraUnavailableLabel.isHidden = true
    self.resumeButton.isHidden = true
  }

  func presentVideoConfigurationErrorAlert(_ manager: CameraFeedManager) {
    let alertController = UIAlertController(
      title: "Confirguration Failed", message: "Configuration of camera has failed.",
      preferredStyle: .alert)
    let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
    alertController.addAction(okAction)

    present(alertController, animated: true, completion: nil)
  }

  func presentCameraPermissionsDeniedAlert(_ manager: CameraFeedManager) {
    let alertController = UIAlertController(
      title: "Camera Permissions Denied",
      message:
        "Camera permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { action in
      if let url = URL.init(string: UIApplication.openSettingsURLString) {
        UIApplication.shared.open(url, options: [:], completionHandler: nil)
      }
    }

    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }

  @objc func runModel(on pixelBuffer: CVPixelBuffer) {
    guard let overlayViewFrame = overlayViewFrame, let previewViewFrame = previewViewFrame
    else {
      return
    }
    // To put `overlayView` area as model input, transform `overlayViewFrame` following transform
    // from `previewView` to `pixelBuffer`. `previewView` area is transformed to fit in
    // `pixelBuffer`, because `pixelBuffer` as a camera output is resized to fill `previewView`.
    // https://developer.apple.com/documentation/avfoundation/avlayervideogravity/1385607-resizeaspectfill
    let modelInputRange = overlayViewFrame.applying(
      previewViewFrame.size.transformKeepAspect(toFitIn: pixelBuffer.size))

    // Run Midas model.
    guard
      let (result, width, height, times) = self.modelDataHandler?.runMidas(
        on: pixelBuffer,
        from: modelInputRange,
        to: overlayViewFrame.size)
    else {
      os_log("Cannot get inference result.", type: .error)
      return
    }

    if avg_latency == 0 {
        avg_latency = times.inference
    } else {
        avg_latency = times.inference*0.1 + avg_latency*0.9
    }
    
    // Udpate `inferencedData` to render data in `tableView`.
    inferencedData = InferencedData(score: Float(avg_latency), times: times)
    
    //let height = 256
    //let width = 256
    
    let outputs = result
    let outputs_size = width * height;
      
    var multiplier : Float = 1.0;
    
    let max_val : Float = outputs.max() ?? 0
    let min_val : Float = outputs.min() ?? 0
    
    if((max_val - min_val) > 0) {
        multiplier = 255 / (max_val - min_val);
    }
    
    // Draw result.
    DispatchQueue.main.async {
      self.tableView.reloadData()
                        
        var pixels: [PixelData] = .init(repeating: .init(a: 255, r: 0, g: 0, b: 0), count: width * height)
             
        for i in pixels.indices {
        //if(i < 1000)
        //{
            let val = UInt8((outputs[i] - min_val) * multiplier)
            
            pixels[i].r = val
            pixels[i].g = val
            pixels[i].b = val
        //}
        }
        
        
        /*
           pixels[i].a = 255
           pixels[i].r = .random(in: 0...255)
           pixels[i].g = .random(in: 0...255)
           pixels[i].b = .random(in: 0...255)
        }
        */
        
        DispatchQueue.main.async {
            let image = UIImage(pixels: pixels, width: width, height: height)

            self.imageView.image = image
            
            if (self.imageViewInitialized == false) {
                self.imageViewInitialized = true
                self.overlayView.addSubview(self.imageView)
                self.overlayView.setNeedsDisplay()
            }
        }
        
        /*
        let image = UIImage(pixels: pixels, width: width, height: height)
        
        var imageView : UIImageView
        imageView  = UIImageView(frame:CGRect(x:0, y:0, width:400, height:400));
        imageView.image = image
        self.overlayView.addSubview(imageView)
        self.overlayView.setNeedsDisplay()
        */
    }
  }
/*
  func drawResult(of result: Result) {
    self.overlayView.dots = result.dots
    self.overlayView.lines = result.lines
    self.overlayView.setNeedsDisplay()
  }

  func clearResult() {
    self.overlayView.clear()
    self.overlayView.setNeedsDisplay()
  }
    */
    
}


// MARK: - TableViewDelegate, TableViewDataSource Methods
extension ViewController: UITableViewDelegate, UITableViewDataSource {
  func numberOfSections(in tableView: UITableView) -> Int {
    return InferenceSections.allCases.count
  }

  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    guard let section = InferenceSections(rawValue: section) else {
      return 0
    }

    return section.subcaseCount
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    let cell = tableView.dequeueReusableCell(withIdentifier: "InfoCell") as! InfoCell
    guard let section = InferenceSections(rawValue: indexPath.section) else {
      return cell
    }
    guard let data = inferencedData else { return cell }

    var fieldName: String
    var info: String

    switch section {
    case .Score:
      fieldName = section.description
      info = String(format: "%.3f", data.score)
    case .Time:
      guard let row = ProcessingTimes(rawValue: indexPath.row) else {
        return cell
      }
      var time: Double
      switch row {
      case .InferenceTime:
        time = data.times.inference
      }
      fieldName = row.description
      info = String(format: "%.2fms", time)
    }

    cell.fieldNameLabel.text = fieldName
    cell.infoLabel.text = info

    return cell
  }

  func tableView(_ tableView: UITableView, heightForRowAt indexPath: IndexPath) -> CGFloat {
    guard let section = InferenceSections(rawValue: indexPath.section) else {
      return 0
    }

    var height = Traits.normalCellHeight
    if indexPath.row == section.subcaseCount - 1 {
      height = Traits.separatorCellHeight + Traits.bottomSpacing
    }
    return height
  }

}

// MARK: - Private enums
/// UI coinstraint values
fileprivate enum Traits {
  static let normalCellHeight: CGFloat = 35.0
  static let separatorCellHeight: CGFloat = 25.0
  static let bottomSpacing: CGFloat = 30.0
}

fileprivate struct InferencedData {
  var score: Float
  var times: Times
}

/// Type of sections in Info Cell
fileprivate enum InferenceSections: Int, CaseIterable {
  case Score
  case Time

  var description: String {
    switch self {
    case .Score:
      return "Average"
    case .Time:
      return "Processing Time"
    }
  }

  var subcaseCount: Int {
    switch self {
    case .Score:
      return 1
    case .Time:
      return ProcessingTimes.allCases.count
    }
  }
}

/// Type of processing times in Time section in Info Cell
fileprivate enum ProcessingTimes: Int, CaseIterable {
  case InferenceTime

  var description: String {
    switch self {
    case .InferenceTime:
      return "Inference Time"
    }
  }
}
