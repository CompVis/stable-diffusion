# My Stable Diffusion
This is my personal fork of Stable Diffusion. The first goal was to get it to run on my PC with sensible defaults (the out-of-the-box defaults don't run because they require more VRAM.) Also minor changes like removing the watermark code and using a more descriptive file name.

The second goal was to be able to run it with a GUI, where the model is loaded only once, thus saving a lot of time. At the same time, prompts are stored to a text file so interesting images can be further iterated upon later on.

The third goal was to create a pipeline to other tools including upscaling and face-fixing.

Note that everything I've done uses the prefix "my-", including the Conda environment which is "my-ldm". To get started, run this (one-time) command:
"conda env create -f environment.yaml".

Make sure the weight file is copied to the appropriate folder. See this guide for basic steps: https://www.assemblyai.com/blog/how-to-run-stable-diffusion-locally-to-generate-images/