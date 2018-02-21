import os
import cv2
import xml.etree.ElementTree as ET
import random


DATA_DIR = '/media/cuong-nguyen/B0EA64A4EA646894/backup_ubuntu/2018/data' # to be customized
this_dir = os.path.dirname(__file__)

def check(obj_name):
    return  obj_name == 'nom' or obj_name == 'prenom' \
        or obj_name == 'datenaissance' or obj_name == 'lieu' \
        or obj_name == 'mrz1' or obj_name=="mrz2" \
        or obj_name == 'mrz' or obj_name=="nomepouse"


def parse_rec(filename):
	""" Parse a PASCAL VOC xml file """
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
	for dir_name in range(3):
		if not os.path.exists(str(dir_name)):
			os.makedirs(str(dir_name))
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
				training_img_name = imagename + '.png'
				#DEST_DIR = 'lieu' if obj_name == 'lieu' else 'nom'
				DEST_DIR = str(dir_name)+"/"+obj_name
				if not os.path.exists(DEST_DIR):
	                		os.makedirs(DEST_DIR)
				training_img_path = os.path.join(this_dir, DEST_DIR, training_img_name)
				is_crop=random.randint(0,1)
				is_crop=1
				crop_size=3
				x_crop=random.randint(1,crop_size)
				y_crop=random.randint(1,crop_size)
				w_crop=random.randint(1,crop_size)
				h_crop=random.randint(1,crop_size)
				stretch_size=3
				x_stretch=random.randint(1,stretch_size)
				y_stretch=random.randint(1,stretch_size)
				w_stretch=random.randint(1,stretch_size)
				w_stretch=5
				h_stretch=random.randint(1,stretch_size)
				
				if(is_crop==0): 
					cv2.imwrite(training_img_path, img[pts[1]+y_crop:pts[3]-h_crop, pts[0]+x_crop:pts[2]-w_crop])
				else:
					cv2.imwrite(training_img_path, img[pts[1]-y_stretch:pts[3]+h_stretch, pts[0]-x_stretch:pts[2]+w_stretch])
print "Finish"
