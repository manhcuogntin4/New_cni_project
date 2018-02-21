import requests
import glob
import sys
import json
import io
import os.path
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="images folder")

args = vars(ap.parse_args())

FOLDER=args["folder"]

def readFileFolder(strFolderName, suffix):
	print strFolderName
	file_list = []
	st=strFolderName+"*."+suffix
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list

def readBaseNameFolder(strFolderName, suffix):
	print strFolderName
	file_list = []
	st=strFolderName+"*."+suffix
	for filename in glob.glob(st): #assuming gif
	    basename=os.path.splitext(os.path.basename(filename))[0]
	    file_list.append(basename)
	return file_list

def removeFileText(ls_imageFile, ls_txtFile):
	for filetext in ls_txtFile:
		if filetext not in ls_imageFile:
			file_name=os.path.join(FOLDER, filetext+".txt")
			os.remove(file_name)
	for fileimage in ls_imageFiles:
		if fileimage not in ls_txtFile:
			file_name=os.path.join(FOLDER, fileimage+".png")
			os.remove(file_name)

def display_lose_file(ls_imageFile, ls_txtFile):
	for filetext in ls_txtFile:
		if filetext not in ls_imageFile:
			print filetext
	for fileimage in ls_imageFiles:
		if fileimage not in ls_txtFile:
			print fileimage

ls_imageFiles=readBaseNameFolder(FOLDER, "png")
print ls_imageFiles
ls_txtFiles=readBaseNameFolder(FOLDER,"txt")
print ls_txtFiles

removeFileText(ls_imageFiles, ls_txtFiles)
#display_lose_file(ls_imageFiles, ls_txtFiles)