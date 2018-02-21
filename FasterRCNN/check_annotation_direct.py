import os
import cv2
import xml.etree.ElementTree as ET
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
    help="images folder")
args = vars(ap.parse_args())

ANNOT_FOLDER=args["folder"]

def readFileFolder(strFolderName):
    print strFolderName
    file_list = []
    st=strFolderName+"/*.xml"
    for filename in glob.glob(st): #assuming gif
        file_list.append(filename)
    return file_list

ls_annots=readFileFolder(ANNOT_FOLDER)

def check(obj_name):
    return  obj_name == 'nom' or obj_name == 'prenom' \
        or obj_name == 'datenaissance' or obj_name == 'lieu' \
        or obj_name == 'person' or obj_name == 'mrz1' or obj_name=="mrz2" \
        or obj_name == 'mrz' or obj_name=="cni"


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    #print filename
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('image/object'):
        if not check(obj.find('name').text):
            continue
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


if __name__ == '__main__':

    cnt_nom, cnt_prenom, cnt_naissance, cnt_cni, cnt_lieu, cnt_mrz, cnt_mrz1, cnt_mrz2, cnt_person= 0, 0, 0, 0, 0, 0, 0,0,0
    for i, annotfile in enumerate(ls_annots):
        #image_file = os.path.join(DATA_DIR, 'Images', imagename + '.png')
        #img = cv2.imread(image_file)
        cnt = {}
        for obj in parse_rec(annotfile):
            obj_name = obj['name']
            pts = obj['bbox']
            if obj_name not in cnt:
                cnt[obj_name] = 0
            else:
                cnt[obj_name] += 1
            if obj_name == 'nom':
                cnt_nom += 1
            if obj_name == 'prenom':
                cnt_prenom += 1
            if obj_name == 'datenaissance':
                cnt_naissance += 1
            if obj_name == 'lieu':
                cnt_lieu += 1
            if obj_name == 'mrz':
                cnt_mrz += 1
            if obj_name == 'mrz1':
                cnt_mrz1 += 1
            if obj_name == 'mrz2':
                cnt_mrz2 += 1
            if obj_name=="person":
                cnt_person+=1
            if obj_name=="cni":
                cnt_cni+=1
        print cnt_nom, cnt_prenom, cnt_naissance, cnt_cni, cnt_lieu, cnt_mrz, cnt_mrz1, cnt_mrz2, cnt_person
        if cnt_nom != cnt_prenom or cnt_prenom != cnt_naissance or cnt_naissance != cnt_lieu or cnt_lieu !=cnt_mrz or cnt_mrz !=cnt_mrz1 \
        or cnt_mrz1 !=cnt_mrz2 or cnt_mrz2!=cnt_person or cnt_person!=cnt_cni:
    		print imagename
            	break
 

