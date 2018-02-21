import os
import cv2
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cStringIO as StringIO
import urllib
import exifutil
from tools.axademo import detect_cni
from tools.axademo_carte_grise import detect_carte_grise

from tools.axademo_permis import detect_permis
from tools.axademo_permis_short import detect_permis_short

from tools.axademo_nouveaupermis import detect_nouveaupermis

from flask import Flask, redirect, url_for, request, session, abort, render_template, flash
import os
import caffe
import glob
import SOAPpy
from io import BytesIO
import requests
import caffe

#Mulitprocess with child process
from multiprocessing.pool import Pool as PoolParent
from multiprocessing import Process
import time
#Process Unicode here
import sys
reload(sys)
sys.setdefaultencoding("ISO-8859-1")

#Process image with textcleaner
import subprocess
#Convert image to string
import base64
import copy
from difflib import SequenceMatcher
# json fie

from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename    
from flask import jsonify
from flask import request

# Remove accent 
from unidecode import unidecode

app = Flask(__name__)


CAFFE_ROOT ='/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
MODELE_DIR = os.path.join(REPO_DIR, 'models/googlenet')
DATA_DIR = os.path.join(REPO_DIR, 'data/googlenet')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


UPLOAD_FOLDER_IMG = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER_IMG'] = UPLOAD_FOLDER_IMG

def allowed_file(filename):
  # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print '**found file', file.filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename))
            # for browser, add 'redirect' function on top of 'url_for'
            file_name=os.path.join(UPLOAD_FOLDER_IMG,filename)
            return classify_image_restful(file_name)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)




class ImagenetClassifier(object):
	default_args = {
		'model_def_file': (
			'{}/deploy.prototxt'.format(MODELE_DIR)),
		'pretrained_model_file': (
			'{}/train_val.caffemodel'.format(DATA_DIR)),
		'mean_file': (
			'{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(CAFFE_ROOT)),
		'class_labels_file': (
			'{}/synset_words.txt'.format(MODELE_DIR)),
		'bet_file': (
			'{}/data/ilsvrc12/imagenet.bet.pickle'.format(CAFFE_ROOT)),
	}
	for key, val in default_args.iteritems():
		if not os.path.exists(val):
			raise Exception(
				"File for {} is missing. Should be at: {}".format(key, val))
	default_args['image_dim'] = 256
	default_args['raw_scale'] = 255.

	def __init__(self, model_def_file, pretrained_model_file, mean_file,
				 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
		logging.info('Loading net and associated files...')
		if gpu_mode:
			caffe.set_mode_gpu()
		else:
			caffe.set_mode_cpu()
		net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)
		self.net = net
		transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))
		# transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
		transformer.set_raw_scale('data', 255)
		transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead of BGR
		self.transformer = transformer

		with open(class_labels_file) as f:
			labels_df = pd.DataFrame([
				{
					'synset_id': l.strip().split(' ')[0],
					'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
				}
				for l in f.readlines()
			])
		self.labels = labels_df.sort('synset_id')['name'].values

	def classify_image(self, image):
		try:
			net = self.net
			net.blobs['data'].data[...] = self.transformer.preprocess('data', image)

			starttime = time.time()
			# scores = self.net.predict([image], oversample=True).flatten()
			out = net.forward()
			proba = out['prob'][0]
			scores = net.blobs['fc8'].data[0]
			endtime = time.time()

			indices = (-proba).argsort()[:5]
			predictions = self.labels[indices]

			# In addition to the prediction text, we will also produce
			# the length for the progress bar visualization.
			meta_proba = [
				(p, '%.5f' % proba[i])
				for i, p in zip(indices, predictions)
			]

			score = [
				(p, '%.5f' % scores[i])
				for i, p in zip(indices, predictions)
			]
			logging.info('result: %s', str(meta_proba))

			return (True, score, meta_proba, '%.3f' % (endtime - starttime))

		except Exception as err:
			logging.info('Classification error: %s', err)
			return (False, 'Something went wrong when classifying the '
						   'image. Maybe try another one?')


def ocr_cartegirse_direct(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	file_out="out.png"
	rc=subprocess.check_call(["./textcleaner", path, file_out])
	p=MyPool(8)
	res =p.map(detect_carte_grise,[path, file_out])
	p.close()
	p.join()
	cnis, preproc_time, roi_file_images=res[0]
	cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[1] 
	res_tmp=[r for i,r,p in cnis_tmp]
	for img, res, pt in cnis:
		bbox, text_info = [], {}
		for cls in res:
			bbox.append(res[cls][0])
			text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
			# Take the process of textcleaner if the result not good
			if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="numero":
				print "not correct"
				if (res_tmp[0][cls][2]>res[cls][2]):
					text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))

	return jsonify({"result":(bbox, text_info)})


def ocr_cni_direct(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	file_out="out.png"
	rc=subprocess.check_call(["./textcleaner", "-u",  path, file_out])
	p=MyPool(8)
	res =p.map(detect_cni,[path])
	p.close()
	p.join()
	cnis, preproc_time, roi_file_images=res[0]
	cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[0] 
	res_tmp=[r for i,r,p in cnis_tmp]
	for img, res, pt in cnis:
		bbox, text_info = [], {}
		for cls in res:
			bbox.append(res[cls][0])
			text_info[cls] = (unidecode(res[cls][1]), '%.3f' % (res[cls][2]))   # (content, prob)
			# Take the process of textcleaner if the result not good
			if res_tmp:
				if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="lieu":
					print "not correct"
					if (res_tmp[0][cls][2]>res[cls][2]):
						text_info[cls] = (unidecode(res_tmp[0][cls][1]), '%.3f' % (res_tmp[0][cls][2]))

	return jsonify({"result":(bbox, text_info)})



def ocr_permis_direct(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	file_out="out.png"
	original_img = cv2.imread(path)
	clone_img = copy.copy(original_img)
	cv2.imwrite(file_out, clone_img)
	p=MyPool(8)
	res =p.apply_async(detect_permis,(path,))
	res1 =p.apply_async(detect_permis_short,(file_out,))
	p.close()
	p.join()
	cnis, preproc_time, roi_file_images=res.get()
	cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res1.get() 
	res_tmp=[r for i,r,p in cnis_tmp]
	res_long=[r for i,r,p in cnis]
	for img, res, pt in cnis:
            #ptime += pt
            #images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
                if check(res[cls][2], cls, res[cls][1]) and (cls in res_tmp[0]):
                #if (cls in res_tmp[0]):
                    if cls=="nom" or cls=="prenom" or len(res[cls][1])==10:
                        if (res_tmp[0][cls][2]>res[cls][2]) and ( res_tmp[0][cls][2]>0.8 or len(res_tmp[0][cls][1])>=len(res[cls][1])):
                        	print "change"
                        	text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))
                    elif cls=="date_naissance" and similar(res_tmp[0][cls][1], res[cls][1])>0.8 and (res_tmp[0][cls][2]>res[cls][2]):
                        text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))
                    else:
                        text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))
                elif cls in res_tmp[0]:
                	if (cls=="nom" or cls=="prenom") and res_tmp[0][cls][2]>0.9 and res[cls][1] in res_tmp[0][cls][1] :
                		print "OK here"
                		text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))
            #bboxes.append(bbox)
            #texts.append(text_info)

	return jsonify({"result":(bbox, text_info)})



def ocr_nouveaupermis_direct(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	file_out="out.png"
	rc=subprocess.check_call(["./textcleaner", path, file_out])
	p=MyPool(8)
	res =p.map(detect_nouveaupermis,[path, path])
	logging.info('Extracting Region of Interest in nouveau_permis detect success...')
	p.close()
	p.join()
	cnis, preproc_time, roi_file_images=res[0]
	cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[1] 
	res_tmp=[r for i,r,p in cnis_tmp]
	print "res_tmp:", res_tmp
	__roi_file_images__= roi_file_images
	for img, res, pt in cnis:
		bbox, text_info = [], {}
		for cls in res:
			bbox.append(res[cls][0])
			text_info[cls] = (unidecode(res[cls][1]), '%.3f' % (res[cls][2]))   # (content, prob)
            
			if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="numero":
				print "not correct"
				if (res_tmp[0][cls][2]>res[cls][2]):
					text_info[cls] = (unidecode(res_tmp[0][cls][1]), '%.3f' % (res_tmp[0][cls][2]))

	return jsonify({"result":(bbox, text_info)})


class NoDaemonProcess(Process):
	def _get_daemon(self):
	    return False
	def _set_daemon(self, value):
	    pass
	daemon = property(_get_daemon, _set_daemon)

class MyPool(PoolParent):
    Process = NoDaemonProcess

def classify_image_restful(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	parser = optparse.OptionParser()
	parser.add_option('-d', '--debug', help="enable debug mode", action="store_true", default=False)
	parser.add_option(
		'-p', '--port',
		help="which port to serve content on",
		type='int', default=5000)
	parser.add_option(
		'-g', '--gpu',
		help="use gpu mode",
		action='store_true', default=False)
	opts, args = parser.parse_args()
	ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})
	classifier=ImagenetClassifier(**ImagenetClassifier.default_args)
	classifier.net.forward()
	result= classifier.classify_image(image)
	print result[2]
	return jsonify({"result":result[2]})

def classify_image(path):
	image = caffe.io.load_image(path)
	try:
		os.mkdir(UPLOAD_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(UPLOAD_FOLDER, 0755)	
	parser = optparse.OptionParser()
	parser.add_option('-d', '--debug', help="enable debug mode", action="store_true", default=False)
	parser.add_option(
		'-p', '--port',
		help="which port to serve content on",
		type='int', default=5000)
	parser.add_option(
		'-g', '--gpu',
		help="use gpu mode",
		action='store_true', default=False)
	opts, args = parser.parse_args()
	ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})
	classifier=ImagenetClassifier(**ImagenetClassifier.default_args)
	classifier.net.forward()
	result= classifier.classify_image(image)
	return result


def check(prob, cls, txt):
    if ((cls=="nom" or cls=="prenom") and prob<0.6) or (cls in ['date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B'] and prob<0.8 ) or \
    ( cls in ['date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B'] and len(txt)!=10)\
    or (cls=="date_naissance" and prob >0.9 ):
        return True
    else:
        return False

def ocr_direct_restful(path):

	result = classify_image(path)	
	proba_list = result[2]  # array of (cls_name, probability)
	is_cni = False # proba_list[0][0] == 'cni'
	is_carte_grise = False
	is_permis=False
	is_nouveaupermis=False
	is_passeport=False
	logging.info('proba_list found %s', proba_list[0][0])

	if proba_list[0][0] == 'cni':
		is_cni=True
	if proba_list[0][0] == 'cartegrise':
		is_carte_grise=True
	if proba_list[0][0]=='permis':
		is_permis=True

	if proba_list[0][0]=='nouveaupermis':
		is_nouveaupermis=True

	if proba_list[0][0]=='passeport':
		is_passeport=True
	if is_cni:
		return ocr_cni_direct(path)
	elif is_carte_grise:
		return ocr_cartegirse_direct(path)
	elif is_permis:
		return ocr_permis_direct(path)
	elif is_nouveaupermis:
		return ocr_nouveaupermis_direct(path)
	elif is_passeport:
		logging.info('Extracting Region of Interest in passport...')
		return "passport"
	else:
		return "not_found_type"


@app.route('/ocr/', methods=['GET', 'POST'])
def upload_file_ocr():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            print '**found file', file.filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename))
            # for browser, add 'redirect' function on top of 'url_for'
            file_name=os.path.join(UPLOAD_FOLDER_IMG,filename)
            return ocr_direct_restful(file_name)


@app.route('/is_alive',methods=['GET','POST'])
def is_alive():
	p = subprocess.Popen(["ps", "-a"], stdout=subprocess.PIPE)
	out, err = p.communicate()
	if ('python2' in str(out)):
	    return "service running"
	else:
	    return "service is not running"

if __name__ == "__main__":
	app.run()