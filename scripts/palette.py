import cv2
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required = True, help = "Path to image file")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

def quant(img,k):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    data = np.float32(image).reshape(-1,3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(data, k , None, criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

if os.path.isfile(args["file"]):
    print(args["file"])
    cv2.imwrite(args["file"], quant(cv2.imread(args["file"]),args["clusters"]))