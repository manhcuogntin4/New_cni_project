import os
import cv2
import xml.etree.ElementTree as ET

#DATA_DIR = '/home/cuong-nguyen/2017/Projets/Permis/Extract-Permis-/Code_sources/Faster_RCNN/axa_permis_dataset/data' # to be customized
DATA_DIR = '/media/cuong-nguyen/B0EA64A4EA646894/backup_ubuntu/2018/data' # to be customized
this_dir = os.path.dirname(__file__)

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
    for obj in tree.findall('object'):
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
    annopath = os.path.join(
            DATA_DIR,
            'Annotations',
            '{:s}.xml')
    imagesetfile = os.path.join(DATA_DIR, 'ImageSets', 'train.txt')
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    cnt_nom, cnt_prenom, cnt_naissance, cnt_cni, cnt_lieu, cnt_mrz, cnt_mrz1, cnt_mrz2, cnt_person= 0, 0, 0, 0, 0, 0, 0,0,0
    for i, imagename in enumerate(imagenames):
        image_file = os.path.join(DATA_DIR, 'Images', imagename + '.png')
        img = cv2.imread(image_file)
        cnt = {}
        for obj in parse_rec(annopath.format(imagename)):
            obj_name = obj['name']
            pts = obj['bbox']
            if obj_name not in cnt:
                cnt[obj_name] = 0
            else:
                cnt[obj_name] += 1
            training_img_name = imagename + '_' + obj_name + str(cnt[obj_name]) + '.png'
            # DEST_DIR = 'lieu' if obj_name == 'lieu' else 'nom'
            DEST_DIR = obj_name
            if not os.path.exists(DEST_DIR):
                os.makedirs(DEST_DIR)
            training_img_path = os.path.join(this_dir, DEST_DIR, training_img_name)
            #cv2.imwrite(training_img_path, img[pts[1]:pts[3], pts[0]:pts[2]])
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
 

