#!/bin/bash
# Download TF Lite model from the internet if it does not exist.

TFLITE_MODEL="model_opt.tflite"
TFLITE_FILE="Midas/Model/${TFLITE_MODEL}"
MODEL_SRC="https://github.com/isl-org/MiDaS/releases/download/v2/${TFLITE_MODEL}"

if test -f "${TFLITE_FILE}"; then
    echo "INFO: TF Lite model already exists. Skip downloading and use the local model."
else
    curl --create-dirs -o "${TFLITE_FILE}" -LJO "${MODEL_SRC}"
    echo "INFO: Downloaded TensorFlow Lite model to ${TFLITE_FILE}."
fi

