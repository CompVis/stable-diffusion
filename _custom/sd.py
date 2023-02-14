#!/usr/bin/env python3

import os
import sys
import time
import random
import re
import hashlib

# Params
seed = 387974712
randomSeed = False
numSamples = 3
numIter = 1
numSteps = 500
width = 64 * 16
height = 64 * 24
# width = 64 * 14
# height = 64 * 16
# width = 64 * 8
# height = 64 * 8

# IMG2IMG params
strength = 0.65
testMode = False

# Mode
modes = ["TXT2IMG", "IMG2IMG", "TXT2IMG_OLD"]
mode = modes[1]

# Script
scriptPath = ''
if mode == modes[0]:
    scriptPath = "optimizedSD/optimized_txt2img.py"
if mode == modes[1]:
    scriptPath = "optimizedSD/optimized_img2img.py"
if mode == modes[2]:
    scriptPath = "scripts/txt2img.py"

# Model Checkpoint
checkpoints = [
    # "sd-v1-4.ckpt",
    # "v1-5-pruned-emaonly.ckpt",
    # "v1-5-pruned.ckpt",
    "dreamshaper_33.ckpt", # Versatile, Anime/Character art
    # "uberRealisticMerge_urpmv12.ckpt", # Photorealistic people
    # "elldrethsLucidMix_v10.ckpt", # Cartoony and saturated
    # "artErosAerosATribute_aerosNovae.ckpt", # Broken
]

# Input image for img2img
inputPath = "/mnt/c/Users/chris/Desktop/stable-diffusion-optimized/input/"
inputImageName = "641049307_00019.png"
inputImage = inputPath + inputImageName

# Join prompt from args
args = sys.argv
prompt = " ".join(args[1:])

# Base out dir
outDir = ''
if mode == modes[0]:
    outDir = "/mnt/c/Users/chris/Desktop/stable-diffusion-optimized/images/"
if mode == modes[1]:
    outDir = "/mnt/c/Users/chris/Desktop/stable-diffusion-optimized/images2images/"
if mode == modes[2]:
    outDir = "/mnt/c/Users/chris/Desktop/stable-diffusion/images/"

# Join Prompt
rootName = re.escape("_".join(args[1:]))
if testMode:
    rootName += "_0"

# Use a truncated description and hash if prompt is too long and write it to a file
if len(rootName) > 25:
    rootName = "_".join(args[1:4]) + "_" + hashlib.md5(bytes(rootName, 'utf-8')).hexdigest()
    os.system("mkdir -p " + outDir + rootName)
    os.system("echo " + prompt + " > " + outDir + rootName + "/prompt.txt")

start = time.time()

iterations = 1

for i in range(iterations):
    # Randomize seed per iteration
    if randomSeed:
        seed = random.randint(0, 4294967295)

    for checkpoint in checkpoints:
        # Build Command
        command = ''
        command += 'python ' + scriptPath
        if mode == modes[2]:
            command += ' --plms --skip_grid'
        else:
            command += ' --turbo'
        command += ' --ckpt ' + checkpoint
        command += ' --n_samples ' + str(numSamples)
        command += ' --n_iter ' + str(numIter)
        command += ' --ddim_steps ' + str(numSteps)
        command += ' --outdir ' + outDir + rootName + "/ckpt_" + checkpoint + "/"
        if mode == modes[1]:
            command += inputImageName + "/"
        command += ' --seed ' + str(seed)
        command += ' --W ' + str(width)
        command += ' --H ' + str(height)
        command += ' --prompt "' + prompt + '"'
        if mode == modes[1]:
            command += ' --strength "' + str(strength) + '"'
            command += ' --init-img "' + inputImage + '"'

        # Fire command
        os.system(command)

    # Touch the dir to bump last update
    os.system("touch " + outDir + rootName)

print('Elapsed: ' + str(time.time()-start) + 's')
