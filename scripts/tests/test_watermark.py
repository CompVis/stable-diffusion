import cv2
import fire
from imwatermark import WatermarkDecoder


def testit(img_path):
    bgr = cv2.imread(img_path)
    decoder = WatermarkDecoder("bytes", 136)
    watermark = decoder.decode(bgr, "dwtDct")
    try:
        dec = watermark.decode("utf-8")
    except:
        dec = "null"
    print(dec)


if __name__ == "__main__":
    fire.Fire(testit)
