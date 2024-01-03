# Tensorflow Lite MiDaS iOS Example

### Requirements

- XCode 11.0 or above
- iOS 12.0 or above, [iOS 14 breaks the NPU Delegate](https://github.com/tensorflow/tensorflow/issues/43339)
- TensorFlow 2.4.0, TensorFlowLiteSwift -> 0.0.1-nightly

## Quick Start with a MiDaS Example

MiDaS is a neural network to compute depth from a single image. It uses TensorFlowLiteSwift / C++ libraries on iOS. The code is written in Swift.

Paper: https://arxiv.org/abs/1907.01341

> Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer
> René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

### Install TensorFlow

Set default python version to python3:

```
echo 'export PATH=/usr/local/opt/python/libexec/bin:$PATH' >> ~/.zshenv
echo 'alias python=python3' >> ~/.zshenv
echo 'alias pip=pip3' >> ~/.zshenv
```

Install TensorFlow

```shell
pip install tensorflow
```

### Install TensorFlowLiteSwift via Cocoapods

Set required TensorFlowLiteSwift version in the file (`0.0.1-nightly` is recommended): https://github.com/isl-org/MiDaS/blob/master/mobile/ios/Podfile#L9

Install: brew, ruby, cocoapods

```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew install mc rbenv ruby-build
sudo gem install cocoapods
```


The TensorFlowLiteSwift library is available in [Cocoapods](https://cocoapods.org/), to integrate it to our project, we can run in the root directory of the project:

```ruby
pod install
```

Now open the `Midas.xcworkspace` file in XCode, select your iPhone device (XCode->Product->Destination->iPhone) and launch it (cmd + R). If everything works well, you should see a real-time depth map from your camera.

### Model

The TensorFlow (TFlite) model `midas.tflite` is in the folder `/Midas/Model`


To use another model, you should convert it from TensorFlow saved-model to TFlite model (so that it can be deployed):

```python
saved_model_export_dir = "./saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_export_dir)    
tflite_model = converter.convert()
open(model_tflite_name, "wb").write("model.tflite")
```

### Setup XCode

* Open directory `.xcworkspace` from the XCode

* Press on your ProjectName (left-top corner) -> change Bundle Identifier to `com.midas.tflite-npu` or something like this (it should be unique)

* select your Developer Team (your should be signed-in by using your AppleID)

* Connect your iPhone (if you want to run it on real device instead of simulator), select your iPhone device (XCode->Product->Destination->iPhone)

* Click in the XCode: Product -> Run

* On your iPhone device go to the: Settings -> General -> Device Management (or Profiles) -> Apple Development -> Trust Apple Development

----

Original repository: https://github.com/isl-org/MiDaS


### Examples:

| ![photo_2020-09-27_17-43-20](https://user-images.githubusercontent.com/4096485/94367804-9610de80-00e9-11eb-8a23-8b32a6f52d41.jpg) | ![photo_2020-09-27_17-49-22](https://user-images.githubusercontent.com/4096485/94367974-7201cd00-00ea-11eb-8e0a-68eb9ea10f63.jpg) | ![photo_2020-09-27_17-52-30](https://user-images.githubusercontent.com/4096485/94367976-729a6380-00ea-11eb-8ce0-39d3e26dd550.jpg) | ![photo_2020-09-27_17-43-21](https://user-images.githubusercontent.com/4096485/94367807-97420b80-00e9-11eb-9dcd-848ad9e89e03.jpg) |
|---|---|---|---|

## LICENSE

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
