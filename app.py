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
import copy
#Multiprocess
from multiprocessing import Pool   
import multiprocessing.pool
import multiprocessing 
import subprocess
from multiprocessing import Manager
from functools import partial


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
# Process similar
from difflib import SequenceMatcher


#CAFFE_ROOT = '/home/ubuntu/caffe'
CAFFE_ROOT ='/home/cuong-nguyen/2016/Workspace/brexia/Septembre/Codesource/caffe-master'
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
MODELE_DIR = os.path.join(REPO_DIR, 'models/googlenet')
DATA_DIR = os.path.join(REPO_DIR, 'data/googlenet') 
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)

__filename__=""
__isnom__=False
__isprenom__=False
__islieu__=False
__isepouse__=False
__roi_file_images__=[]


@app.route('/login')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('admin.html')
 
@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'password' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else:
        flash('wrong password!')
    return home()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return index()

@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=imageurl,
        is_cni=False)


def load_image(filename, color=True):
    img = cv2.imread(filename).astype(np.float32) / 255.
    img = img[...,::-1]
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        global __filename__
        global __isnom__
        global __isprenom__
        global __islieu__
        global __isepouse__
        global __roi_file_images__
        __isnom__=False
        __isprenom__=False
        __islieu__=False
        __isepouse__=False
        #__filename__=imagefile.filename
        logging.info('result: %s', __filename__)
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
        __filename__=filename
        # image = exifutil.open_oriented_im(filename)
        image = load_image(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    result = app.clf.classify_image(image)
    logging.info('result: %s', result)
    #print result
    proba_list = result[2]  # array of (cls_name, probability)
    print proba_list
   
    is_cni = False # proba_list[0][0] == 'cni'
    is_carte_grise = False
    is_permis=False
    is_nouveaupermis=False
    is_passeport=False
    logging.info('proba_list found %s', proba_list[0][0])
    if proba_list[0][0] == 'cni':
        is_cni=True
        logging.info('cni')
    if proba_list[0][0] == 'cartegrise':
        is_carte_grise=True
        logging.info('carte grise')
    # for permis
    if proba_list[0][0]=='permis':
        is_permis=True
        logging.info('permis')


    if proba_list[0][0]=='nouveaupermis':
    	is_nouveaupermis=True

    if proba_list[0][0]=='passeport':
        is_passeport=True
        logging.info('passport')
        
    ptime = 0.0
    images = []
    images.append(image) # for classification display
    bboxes, texts = [], []
    roi_file_images=[]
    if is_cni:
        logging.info('Extracting Region of Interest in CNI...')
        file_out="out.png"
        rc=subprocess.check_call(["./textcleaner", "-u", filename, file_out])
        p=MyPool(8)
        #res =p.map(detect_cni,[filename, file_out])
        res =p.map(detect_cni,[filename])
        p.close()
        p.join()
        cnis, preproc_time, roi_file_images=res[0]
        cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[0] 
        res_tmp=[r for i,r,p in cnis_tmp]
        __roi_file_images__= roi_file_images
        ptime += preproc_time+ preproc_time_tmp
        for img, res, pt in cnis:
            ptime += pt
            images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
                if cls=="nom":
                    __isnom__=True
                if cls=="prenom":
                    __isprenom__=True
                if cls=="lieu":
                    __islieu__=True
                if cls=="epouse":
                    __isepouse__=True
                #Process textcleaner if result is not good
                if res_tmp:
                    if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="lieu":
                    	if (res_tmp[0][cls][2]>res[cls][2]):
                    		text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))


            bboxes.append(bbox)
            texts.append(text_info)


    elif is_carte_grise:
        logging.info('Extracting Region of Interest in Carte Grise...')
        file_out="out.png"
        rc=subprocess.check_call(["./textcleaner", filename, file_out])
        p=MyPool(8)
        res =p.map(detect_carte_grise,[filename, file_out])
        p.close()
        p.join()
        cnis, preproc_time, roi_file_images=res[0]
        cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[1] 
        res_tmp=[r for i,r,p in cnis_tmp]
        print "res_tmp:", res_tmp
        __roi_file_images__= roi_file_images
        ptime += preproc_time + preproc_time_tmp
        for img, res, pt in cnis:
            ptime += pt
            images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
                
                if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="numero":
                	print "not correct"
                	if (res_tmp[0][cls][2]>res[cls][2]):
                		text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))

            bboxes.append(bbox)
            texts.append(text_info)

    # For permis

    elif is_permis:
        logging.info('Extracting Region of Interest in permis...')
        file_out="out.png"
        original_img = cv2.imread(filename)
        clone_img = copy.copy(original_img)
        p=MyPool(8)
        res =p.apply_async(detect_permis,(filename,))
        res1 =p.apply_async(detect_permis_short,(file_out,))
        p.close()
        p.join()
        cnis, preproc_time, roi_file_images=res.get()
        cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res1.get() 
        res_tmp=[r for i,r,p in cnis_tmp]
        print "res_tmp:", res_tmp

        res_long=[r for i,r,p in cnis]
        print "restlong:", res_long
        __roi_file_images__= roi_file_images
        ptime += preproc_time + preproc_time_tmp
        for img, res, pt in cnis:
            ptime += pt
            images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
                if check(res[cls][2], cls, res[cls][1]) and (cls in res_tmp[0]):
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
            bboxes.append(bbox)
            texts.append(text_info)
    elif is_nouveaupermis:
    	logging.info('Extracting Region of Interest in nouveau_permis...')
        file_out="out.png"
        rc=subprocess.check_call(["./textcleaner", filename, file_out])
        p=MyPool(8)
        res =p.map(detect_nouveaupermis,[filename, filename])
        logging.info('Extracting Region of Interest in nouveau_permis detect success...')
        p.close()
        p.join()
        cnis, preproc_time, roi_file_images=res[0]
        cnis_tmp, preproc_time_tmp, roi_file_images_tmp=res[1] 
        res_tmp=[r for i,r,p in cnis_tmp]
        print "res_tmp:", res_tmp
        __roi_file_images__= roi_file_images
        ptime += preproc_time + preproc_time_tmp
        for img, res, pt in cnis:
            ptime += pt
            images.append(img)
            bbox, text_info = [], {}
            for cls in res:
                bbox.append(res[cls][0])
                text_info[cls] = (res[cls][1], '%.3f' % (res[cls][2]))   # (content, prob)
                
                if(res[cls][2]<0.8) and (cls in res_tmp[0]) and cls !="numero":
                    print "not correct"
                    if (res_tmp[0][cls][2]>res[cls][2]):
                        text_info[cls] = (res_tmp[0][cls][1], '%.3f' % (res_tmp[0][cls][2]))

            bboxes.append(bbox)
            texts.append(text_info)

    elif is_passeport:
        logging.info('Extracting Region of Interest in passport...')

    else:
        images.append(image) ## for future use, just to avoid bug now

    print 'detection ended'
    if not session.get('logged_in'):
        if not roi_file_images:
            for im in roi_file_images:
                os.remove(im)
                it=im+"txt"
                os.remove(it)

        return flask.render_template(
        'index.html', has_result=is_cni or is_carte_grise or is_permis or is_passeport or is_nouveaupermis, result=result, doc_info=texts, proc_time='%.3f' % (ptime),
        imagesrc=embed_image_html(images, bboxes), is_cni=True
        )
    else:
         return flask.render_template(
        'admin.html', has_result=True, result=result, doc_info=texts, proc_time='%.3f' % (ptime),
        imagesrc=embed_image_html(images, bboxes), is_cni=is_cni
    )
    # return flask.render_template(
    #     'index.html', has_result=True, result=result, doc_info=texts, proc_time='%.3f' % (ptime),
    #     imagesrc=embed_image_html(images, bboxes), is_cni=is_cni
    # )

def embed_image_html(images, bboxes):
    """Creates an image embedded in HTML base64 format."""
    embed_images = []
    for im_ind, image in enumerate(images):        
        if im_ind == 0:
            image_pil = Image.fromarray((255 * image).astype('uint8'))
            image_pil = image_pil.resize((256, 256))
            string_buf = StringIO.StringIO()
            image_pil.save(string_buf, format='png')
            data = string_buf.getvalue().encode('base64').replace('\n', '')
            embed_images.append('data:image/png;base64,' + data)
        else:
            if(len(bboxes)>=1):
                for bbox in bboxes[im_ind - 1]:
                    img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    image_pil = Image.fromarray(img.astype('uint8'))
                    string_buf = StringIO.StringIO()
                    image_pil.save(string_buf, format='png')
                    data = string_buf.getvalue().encode('base64').replace('\n', '')
                    embed_images.append('data:image/png;base64,' + data)
       
    return embed_images


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


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


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
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

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)



@app.route('/classify_update', methods=['POST'])
def classify_update():
    nom_change =""
    prenom_change=""
    lieu_change=""
    epouse_change =""

    if __isnom__:
        nom_change = request.form['nom']
    if __isprenom__:
        prenom_change=request.form['prenom']
    if __islieu__:
        lieu_change=request.form['lieu']
    if __isepouse__:
        epouse_change = request.form['epouse']
    print nom_change, prenom_change, lieu_change, epouse_change
    for filename in __roi_file_images__:
        if filename.find('outputnom')!=-1:
                nom=filename+"txt"
                if(nom_change!=""):
                    f_nom=open(nom, "w")
                    f_nom.write(nom_change.encode('utf8'))
                    f_nom.close()
        if filename.find('outputprenom')!=-1:
                prenom=filename+"txt"
                if(prenom_change!=""):
                    f_prenom=open(prenom,"w")
                    f_prenom.write(prenom_change.encode('utf8'))
                    f_prenom.close()
        if filename.find("lieu")!=-1:
                lieu=filename+"txt"
                if(lieu_change!=""):
                    f_lieu=open(lieu,"w")
                    f_lieu.write(lieu_change.encode('utf8'))
                    f_lieu.close()
        if filename.find("epouse")!=-1:
                epouse=filename+"txt"
                if(epouse_change!=""):
                    f_epouse=open(epouse,"w")
                    f_epouse.write(epouse_change.encode('utf8'))
                    f_epouse.close()
    
        
    return flask.render_template('admin.html', has_result=False)

def check(prob, cls, txt):
    if ((cls=="nom" or cls=="prenom") and prob<0.6) or (cls in ['date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B'] and prob<0.8 ) or \
    ( cls in ['date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B'] and len(txt)!=10)\
    or (cls=="date_naissance" and prob >0.9 ):
        return True
    else:
        return False

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def hasXpath(xpath):
    try:
        self.browser.find_element_by_xpath(xpath)
        return True
    except:
        return False

def do_work(queue, filename):
    result=detect_cni(filename)
    print "do_work"
    queue.put(result)

#Multiprocess
class NoDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(PoolParent):
    Process = NoDaemonProcess


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
