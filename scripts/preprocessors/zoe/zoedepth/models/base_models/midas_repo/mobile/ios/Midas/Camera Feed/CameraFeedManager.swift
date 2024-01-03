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

// MARK: - CameraFeedManagerDelegate Declaration
@objc protocol CameraFeedManagerDelegate: class {
  /// This method delivers the pixel buffer of the current frame seen by the device's camera.
  @objc optional func cameraFeedManager(
    _ manager: CameraFeedManager, didOutput pixelBuffer: CVPixelBuffer
  )

  /// This method initimates that a session runtime error occured.
  func cameraFeedManagerDidEncounterSessionRunTimeError(_ manager: CameraFeedManager)

  /// This method initimates that the session was interrupted.
  func cameraFeedManager(
    _ manager: CameraFeedManager, sessionWasInterrupted canResumeManually: Bool
  )

  /// This method initimates that the session interruption has ended.
  func cameraFeedManagerDidEndSessionInterruption(_ manager: CameraFeedManager)

  /// This method initimates that there was an error in video configurtion.
  func presentVideoConfigurationErrorAlert(_ manager: CameraFeedManager)

  /// This method initimates that the camera permissions have been denied.
  func presentCameraPermissionsDeniedAlert(_ manager: CameraFeedManager)
}

/// This enum holds the state of the camera initialization.
// MARK: - Camera Initialization State Enum
enum CameraConfiguration {
  case success
  case failed
  case permissionDenied
}

/// This class manages all camera related functionalities.
// MARK: - Camera Related Functionalies Manager
class CameraFeedManager: NSObject {
  // MARK: Camera Related Instance Variables
  private let session: AVCaptureSession = AVCaptureSession()

  private let previewView: PreviewView
  private let sessionQueue = DispatchQueue(label: "sessionQueue")
  private var cameraConfiguration: CameraConfiguration = .failed
  private lazy var videoDataOutput = AVCaptureVideoDataOutput()
  private var isSessionRunning = false

  // MARK: CameraFeedManagerDelegate
  weak var delegate: CameraFeedManagerDelegate?

  // MARK: Initializer
  init(previewView: PreviewView) {
    self.previewView = previewView
    super.init()

    // Initializes the session
    session.sessionPreset = .high
    self.previewView.session = session
    self.previewView.previewLayer.connection?.videoOrientation = .portrait
    self.previewView.previewLayer.videoGravity = .resizeAspectFill
    self.attemptToConfigureSession()
  }

  // MARK: Session Start and End methods

  /// This method starts an AVCaptureSession based on whether the camera configuration was successful.
  func checkCameraConfigurationAndStartSession() {
    sessionQueue.async {
      switch self.cameraConfiguration {
      case .success:
        self.addObservers()
        self.startSession()
      case .failed:
        DispatchQueue.main.async {
          self.delegate?.presentVideoConfigurationErrorAlert(self)
        }
      case .permissionDenied:
        DispatchQueue.main.async {
          self.delegate?.presentCameraPermissionsDeniedAlert(self)
        }
      }
    }
  }

  /// This method stops a running an AVCaptureSession.
  func stopSession() {
    self.removeObservers()
    sessionQueue.async {
      if self.session.isRunning {
        self.session.stopRunning()
        self.isSessionRunning = self.session.isRunning
      }
    }

  }

  /// This method resumes an interrupted AVCaptureSession.
  func resumeInterruptedSession(withCompletion completion: @escaping (Bool) -> Void) {
    sessionQueue.async {
      self.startSession()

      DispatchQueue.main.async {
        completion(self.isSessionRunning)
      }
    }
  }

  /// This method starts the AVCaptureSession
  private func startSession() {
    self.session.startRunning()
    self.isSessionRunning = self.session.isRunning
  }

  // MARK: Session Configuration Methods.
  /// This method requests for camera permissions and handles the configuration of the session and stores the result of configuration.
  private func attemptToConfigureSession() {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      self.cameraConfiguration = .success
    case .notDetermined:
      self.sessionQueue.suspend()
      self.requestCameraAccess(completion: { granted in
        self.sessionQueue.resume()
      })
    case .denied:
      self.cameraConfiguration = .permissionDenied
    default:
      break
    }

    self.sessionQueue.async {
      self.configureSession()
    }
  }

  /// This method requests for camera permissions.
  private func requestCameraAccess(completion: @escaping (Bool) -> Void) {
    AVCaptureDevice.requestAccess(for: .video) { (granted) in
      if !granted {
        self.cameraConfiguration = .permissionDenied
      } else {
        self.cameraConfiguration = .success
      }
      completion(granted)
    }
  }

  /// This method handles all the steps to configure an AVCaptureSession.
  private func configureSession() {
    guard cameraConfiguration == .success else {
      return
    }
    session.beginConfiguration()

    // Tries to add an AVCaptureDeviceInput.
    guard addVideoDeviceInput() == true else {
      self.session.commitConfiguration()
      self.cameraConfiguration = .failed
      return
    }

    // Tries to add an AVCaptureVideoDataOutput.
    guard addVideoDataOutput() else {
      self.session.commitConfiguration()
      self.cameraConfiguration = .failed
      return
    }

    session.commitConfiguration()
    self.cameraConfiguration = .success
  }

  /// This method tries to an AVCaptureDeviceInput to the current AVCaptureSession.
  private func addVideoDeviceInput() -> Bool {
    /// Tries to get the default back camera.
    guard
      let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
      else {
        fatalError("Cannot find camera")
    }

    do {
      let videoDeviceInput = try AVCaptureDeviceInput(device: camera)
      if session.canAddInput(videoDeviceInput) {
        session.addInput(videoDeviceInput)
        return true
      } else {
        return false
      }
    } catch {
      fatalError("Cannot create video device input")
    }
  }

  /// This method tries to an AVCaptureVideoDataOutput to the current AVCaptureSession.
  private func addVideoDataOutput() -> Bool {
    let sampleBufferQueue = DispatchQueue(label: "sampleBufferQueue")
    videoDataOutput.setSampleBufferDelegate(self, queue: sampleBufferQueue)
    videoDataOutput.alwaysDiscardsLateVideoFrames = true
    videoDataOutput.videoSettings = [
      String(kCVPixelBufferPixelFormatTypeKey): kCMPixelFormat_32BGRA
    ]

    if session.canAddOutput(videoDataOutput) {
      session.addOutput(videoDataOutput)
      videoDataOutput.connection(with: .video)?.videoOrientation = .portrait
      return true
    }
    return false
  }

  // MARK: Notification Observer Handling
  private func addObservers() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(CameraFeedManager.sessionRuntimeErrorOccured(notification:)),
      name: NSNotification.Name.AVCaptureSessionRuntimeError, object: session)
    NotificationCenter.default.addObserver(
      self, selector: #selector(CameraFeedManager.sessionWasInterrupted(notification:)),
      name: NSNotification.Name.AVCaptureSessionWasInterrupted, object: session)
    NotificationCenter.default.addObserver(
      self, selector: #selector(CameraFeedManager.sessionInterruptionEnded),
      name: NSNotification.Name.AVCaptureSessionInterruptionEnded, object: session)
  }

  private func removeObservers() {
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionRuntimeError, object: session)
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionWasInterrupted, object: session)
    NotificationCenter.default.removeObserver(
      self, name: NSNotification.Name.AVCaptureSessionInterruptionEnded, object: session)
  }

  // MARK: Notification Observers
  @objc func sessionWasInterrupted(notification: Notification) {
    if let userInfoValue = notification.userInfo?[AVCaptureSessionInterruptionReasonKey]
      as AnyObject?,
      let reasonIntegerValue = userInfoValue.integerValue,
      let reason = AVCaptureSession.InterruptionReason(rawValue: reasonIntegerValue)
    {
      os_log("Capture session was interrupted with reason: %s", type: .error, reason.rawValue)

      var canResumeManually = false
      if reason == .videoDeviceInUseByAnotherClient {
        canResumeManually = true
      } else if reason == .videoDeviceNotAvailableWithMultipleForegroundApps {
        canResumeManually = false
      }

      delegate?.cameraFeedManager(self, sessionWasInterrupted: canResumeManually)

    }
  }

  @objc func sessionInterruptionEnded(notification: Notification) {
    delegate?.cameraFeedManagerDidEndSessionInterruption(self)
  }

  @objc func sessionRuntimeErrorOccured(notification: Notification) {
    guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError else {
      return
    }

    os_log("Capture session runtime error: %s", type: .error, error.localizedDescription)

    if error.code == .mediaServicesWereReset {
      sessionQueue.async {
        if self.isSessionRunning {
          self.startSession()
        } else {
          DispatchQueue.main.async {
            self.delegate?.cameraFeedManagerDidEncounterSessionRunTimeError(self)
          }
        }
      }
    } else {
      delegate?.cameraFeedManagerDidEncounterSessionRunTimeError(self)
    }
  }
}

/// AVCaptureVideoDataOutputSampleBufferDelegate
extension CameraFeedManager: AVCaptureVideoDataOutputSampleBufferDelegate {
  /// This method delegates the CVPixelBuffer of the frame seen by the camera currently.
  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {

    // Converts the CMSampleBuffer to a CVPixelBuffer.
    let pixelBuffer: CVPixelBuffer? = CMSampleBufferGetImageBuffer(sampleBuffer)

    guard let imagePixelBuffer = pixelBuffer else {
      return
    }

    // Delegates the pixel buffer to the ViewController.
    delegate?.cameraFeedManager?(self, didOutput: imagePixelBuffer)
  }
}
