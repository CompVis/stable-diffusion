# Flex ops are included in the nightly build of the TensorFlow Python package. You can use TFLite models containing Flex ops by the same Python API as normal TFLite models. The nightly TensorFlow build can be installed with this command:
# Flex ops will be added to the TensorFlow Python package's and the tflite_runtime package from version 2.3 for Linux and 2.4 for other environments.
# https://www.tensorflow.org/lite/guide/ops_select#running_the_model

# You must use: tf-nightly
# pip install tf-nightly

import os
import glob
import cv2
import numpy as np

import tensorflow as tf

width=256
height=256
model_name="model.tflite"
#model_name="model_quant.tflite"
image_name="dog.jpg"

# input
img = cv2.imread(image_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
img = (img - mean) / std

img_resized = tf.image.resize(img, [width,height], method='bicubic', preserve_aspect_ratio=False)
#img_resized = tf.transpose(img_resized, [2, 0, 1])
img_input = img_resized.numpy()
reshape_img = img_input.reshape(1,width,height,3)
tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

# load model
print("Load model...")
interpreter = tf.lite.Interpreter(model_path=model_name)
print("Allocate tensor...")
interpreter.allocate_tensors()
print("Get input/output details...")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Get input shape...")
input_shape = input_details[0]['shape']
print(input_shape)
print(input_details)
print(output_details)
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print("Set input tensor...")
interpreter.set_tensor(input_details[0]['index'], tensor)

print("invoke()...")
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
print("get output tensor...")
output = interpreter.get_tensor(output_details[0]['index'])
#output = np.squeeze(output)
output = output.reshape(width, height)
#print(output)
prediction = np.array(output)
print("reshape prediction...")
prediction = prediction.reshape(width, height)
             
# output file
#prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(" Write image to: output.png")
depth_min = prediction.min()
depth_max = prediction.max()
img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
print("save output image...")
cv2.imwrite("output.png", img_out)
        
print("finished")