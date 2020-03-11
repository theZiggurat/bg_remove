import cv2
import numpy as np
import sys
import pathlib
import os
from os import listdir, path
from os.path import isfile, join

BLUR = 25
CANNY_THRESH_1 = 30
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (1.0,0.0,0.0) 

def extract_foreground(img_file, bgpath):
    img = cv2.imread(img_file)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (15,15), 0)

    edges = cv2.Canny(img, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    

    mask_stack  = mask_stack.astype('float32') / 255.0          
    img         = img.astype('float32') / 255.0                 

    bgs = [f for f in listdir(bgpath) if isfile(join(bgpath, f))]
    for bg in bgs:
        background = cv2.imread(join(bgpath, bg))
        dim = (img.shape[1], img.shape[0])
        background = cv2.resize(background, dim, interpolation=cv2.INTER_AREA)
        background = background.astype('float32') / 255.0
        masked = (mask_stack * img) + ((1-mask_stack) * background)
        out = (masked * 255).astype('uint8')  
        out_path = os.path.splitext(img_file)[0]+'--'+bg
        cv2.imwrite(out_path, out)                  


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
bgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[2])
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for f in onlyfiles:
    extract_foreground(join(path,f), bgpath)
