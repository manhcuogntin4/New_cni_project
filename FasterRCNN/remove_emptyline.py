import glob
import sys


IMAGES_FOLDER="./"

def readFileFolder(strFolderName):
	print strFolderName
	file_list = []
	st=strFolderName+"*.xml"
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list

def remove_empty_line(filename):
	with open(filename) as xmlfile:
	    lines = [line for line in xmlfile if line.strip() is not ""]

	with open(filename, "w") as xmlfile:
	    xmlfile.writelines(lines)

ls_files=readFileFolder(IMAGES_FOLDER)

for filename in ls_files:
	remove_empty_line(filename)
