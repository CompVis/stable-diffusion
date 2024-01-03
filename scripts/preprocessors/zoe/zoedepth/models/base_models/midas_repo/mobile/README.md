## Mobile version of MiDaS for iOS / Android - Monocular Depth Estimation

### Accuracy

* Old small model - ResNet50 default-decoder 384x384
* New small model - EfficientNet-Lite3 small-decoder 256x256

**Zero-shot error** (the lower - the better):

| Model |  DIW WHDR | Eth3d AbsRel | Sintel AbsRel | Kitti δ>1.25 | NyuDepthV2 δ>1.25 | TUM δ>1.25 |
|---|---|---|---|---|---|---|
| Old small model 384x384 | **0.1248** | 0.1550 | **0.3300** | **21.81** | 15.73 | 17.00 |
| New small model 256x256 | 0.1344 | **0.1344** | 0.3370 | 29.27 | **13.43** | **14.53** |
| Relative improvement, % | -8 % | **+13 %** | -2 % | -34 % | **+15 %** | **+15 %** |

None of Train/Valid/Test subsets of datasets (DIW, Eth3d, Sintel, Kitti, NyuDepthV2, TUM) were not involved in Training or Fine Tuning.

### Inference speed (FPS) on iOS / Android

**Frames Per Second** (the higher - the better):

| Model | iPhone CPU | iPhone GPU | iPhone NPU | OnePlus8 CPU | OnePlus8 GPU | OnePlus8 NNAPI |
|---|---|---|---|---|---|---|
| Old small model 384x384 | 0.6 | N/A | N/A | 0.45 | 0.50 | 0.50 |
| New small model 256x256 | 8 | 22 | **30** | 6 | **22** | 4 |
| SpeedUp, X times | **12.8x** | - | - | **13.2x** | **44x** | **8x** |

N/A - run-time error (no data available)


#### Models:

* Old small model - ResNet50 default-decoder 1x384x384x3, batch=1 FP32 (converters: Pytorch -> ONNX - [onnx_tf](https://github.com/onnx/onnx-tensorflow) -> (saved model) PB -> TFlite)

    (Trained on datasets: RedWeb, MegaDepth, WSVD, 3D Movies, DIML indoor)

* New small model - EfficientNet-Lite3 small-decoder 1x256x256x3, batch=1 FP32 (custom converter: Pytorch -> TFlite)

    (Trained on datasets: RedWeb, MegaDepth, WSVD, 3D Movies, DIML indoor, HRWSI, IRS, TartanAir, BlendedMVS, ApolloScape)

#### Frameworks for training and conversions:
```
pip install torch==1.6.0 torchvision==0.7.0
pip install tf-nightly-gpu==2.5.0.dev20201031 tensorflow-addons==0.11.2 numpy==1.18.0
git clone --depth 1 --branch v1.6.0 https://github.com/onnx/onnx-tensorflow
```

#### SoC - OS - Library:

* iPhone 11 (A13 Bionic) - iOS 13.7 - TensorFlowLiteSwift 0.0.1-nightly
* OnePlus 8 (Snapdragon 865) - Andoird 10 - org.tensorflow:tensorflow-lite-task-vision:0.0.0-nightly


### Citation

This repository contains code to compute depth from a single image. It accompanies our [paper](https://arxiv.org/abs/1907.01341v3):

>Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer  
René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun

Please cite our paper if you use this code or any of the models:
```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

