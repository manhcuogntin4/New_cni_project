import cv2
import numpy as np
import glob
import imutils
import os
import sys
from string import Template
import subprocess
import argparse
from find_subfolder import *

#IMAGES_FOLDER="./"

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="images folder")

args = vars(ap.parse_args())

IMAGES_FOLDER=args["folder"]



def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"/*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list





def crop_image(filename):
	img=cv2.imread(filename,0)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	res = cv2.resize(img,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	if width>60:
		img_crop=img[:,25:]
		cv2.imwrite(filename, img_crop)


def convert_to_binary(filename):
	img=cv2.imread(filename,0)
	#if (img.shape >= 3):
		#img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	res = cv2.fastNlMeansDenoising(res,20, 7, 21)
	out_path=filename
	cv2.imwrite(out_path,res)

def clean_images(ls_images):
	for filename in ls_images:
		rc=subprocess.check_call(["./textcleaner", "-u", filename, filename])

def crop_images(ls_images):
	for filename in ls_images:
		crop_image(filename)

def covert_images(ls_images):
	for filename in ls_images:
		convert_to_binary(filename)


for subfolder in get_list_subdirectory(IMAGES_FOLDER):
	print subfolder
	ls_images=readFileImages(subfolder)
	print ls_images
	covert_images(ls_images)

#clean_images(ls_images)
#crop_images(ls_images)
