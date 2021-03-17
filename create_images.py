import argparse
import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Images folder path")
ap.add_argument("-f", "--folder", required=True, help="Folder to save potential images")

args = vars(ap.parse_args())

if not os.path.exists(args['folder']):
    os.makedirs(args['folder'])

for folder in os.listdir(args['images']):
    dir = os.path.join(args['images'], folder)
    if os.path.isdir(dir):
        for image in os.listdir(dir):
            if image.endswith('.tif'):
                path = os.path.join(dir, image)
                print(path)
                im = Image.open(path)
                img =  np.array(im)
                cv2.imwrite(args['folder']+"/"+folder+"_"+image[:-4]+'.jpg',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
