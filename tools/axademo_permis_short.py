#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Copyright (c) 2016 Haoming Wang
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from ocr.clstm_permis import clstm_ocr_permis, clstm_ocr_permis_parallel, clstm_ocr_calib_permis, checkdate, checkdateregex, get_similar
#from ocr.clstm_permis import clstm_ocr_calib_carte_grise
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import werkzeug
import datetime
import math
import pytesseract
import re
from PIL import Image

#Multiprocess
from multiprocessing import Pool   
import multiprocessing 
import subprocess
from multiprocessing import Manager
from functools import partial
import multiprocessing.pool
import Queue


CLASSES = ('__background__', # always index 0
                         'carte', 'nom', 'prenom', 'date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'inria': ('INRIA_Person',
                  'INRIA_Person_faster_rcnn_final.caffemodel'),
        'axa': ('axa_poc_permis',
                  'permis_shortdate_shortname.caffemodel')}
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def extract_roi(class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    regions = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return regions

    ind_maxp, maxp = 0, 0
    for i in inds:
        bbox = dets[i, :4]
        print bbox
        score = dets[i, -1]
        if score > maxp:
            maxp = score
            ind_maxp = i

        # a small regulation of detected zone, comment me if the lastest result is good enough
        hight = bbox[3] - bbox[1]
        width= bbox[2] - bbox[0]
        # if class_name == 'nom':

        bbox[0] -= 0.2 * hight
        #     bbox[1] -= 0.05 * hight
        #     bbox[2] += 0.6 * (bbox[2] - bbox[0])
        #     bbox[3] += 0.05 * hight
        if class_name == 'nom':
            # if(width>3*hight):
            #     bbox[0] += 1.7 * hight
            # else:
            #     bbox[0]+=0.3*width
            bbox[1] += 0.05 * hight
            bbox[2] += 0.05  * (bbox[2] - bbox[0])
            bbox[3] += 0.07 * hight

        elif class_name == 'prenom':
            # if(width>3*hight):
            #     bbox[0] += 1.7 * hight
            # else:
            #     bbox[0]+=0.3*width
            bbox[1] -=  0.07* hight
            bbox[2] += 0.05  * (bbox[2] - bbox[0])
            bbox[3] += 0.05 * hight
        else:
            bbox[1] -=  0.07* hight
            #bbox[2] += 0.07  * (bbox[2] - bbox[0])
            bbox[2] += 0.10  * (bbox[2] - bbox[0])
            bbox[3] += 0.07 * hight
    


        pts = [int(bx) for bx in bbox]
        regions.append(pts)
    return [regions[ind_maxp]]

def ocr_queue(im, bbx, cls, q):
    q.put(calib_roi(im,bbx,cls))   

def demo_parallel(net, image_name):
    print "demo parallel"
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(UPLOAD_FOLDER, 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.1#CONF_THRESH = 0.1
    NMS_THRESH = 0.3
    res = {}
    roi_file_name=[]
    # Process msz for check numero
    msz_numero=""
    #Process small size prenom
    bbx_small_prenom=[0,0,0,0] 
    #Parallel processing
    
    list_bbx={}
    list_cls={}
    q={}
    p={}
    pts_tmp={}
    txt_tmp={}
    prob_tmp={}
    for cls_ind, cls in enumerate(CLASSES[2:]):
        cls_ind += 2 # because we skipped background, 'carte'
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        tmp = extract_roi(cls, dets, thresh=CONF_THRESH)
        print len(tmp), cls
        if len(tmp) > 0:#or cls=='ville' or cls=='marque') :
            bbx = tmp[0]  # TODO: Find the zone with greatest probability
            if(bbx[0]<0):
                bbx[0]=0
            if(bbx[2]<0):
                bbx[2]=0
            if(cls!="mrz"):
                list_bbx[cls]=bbx
                list_cls[cls]=cls
            

    for cls in list_cls:
        print "Class:=", cls
        q[cls] = multiprocessing.Queue()
        p[cls] = multiprocessing.Process(target=ocr_queue, args=(im,list_bbx[cls],cls,q[cls],))
        p[cls].start()

    

    # while 1:
    #     running = any(pi.is_alive() for pi in p)
    #     while not queue.empty():
    #         process_queue_data()
    #         if not running:
    #             break

    
    for cls in list_cls:
        p[cls].join()

    for cls in list_cls:
        txt_tmp[cls],prob_tmp[cls]=q[cls].get()
        res[cls] = (list_bbx[cls], txt_tmp[cls], prob_tmp[cls]) 

 
  
    
    im = im[:, :, (2, 1, 0)]
    #return (im, res, timer.total_time, roi_file_name)
    return (im, res, timer.total_time), roi_file_name


def check(boxes, scores, thresh=0.2, nms_thresh=0.3):
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_thresh)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= thresh)[0]
        if cls_ind == 5:  #we skip 'nom epouse' check
            continue
        if len(inds) == 0:
            return False
    return False

def check_numero(numero_clstm, numero_msz):
    print "check_numero"
    print (numero_clstm.find("-")==-1), len(numero_clstm), len(numero_msz)>0, numero_msz.find("-"), re.search('-[0-9][0-9][0-9]-', numero_msz) is not None, re.search('-[0-9][0-9][0-9]-', numero_clstm) is None 
    if (len(numero_clstm)>0 and len(numero_msz)>0 and numero_msz.find("-") and \
        re.search('-[0-9][0-9][0-9]-', numero_clstm) is None and  re.search('-[0-9][0-9][0-9]-', numero_msz) is not None):
        print "numero not correct"
        return True
    else:
        return False
"""demo2 is a complement for demo, in considering the multi-cni case 
    and if we should do faster-rcnn a second time"""
def demo2(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()

    print "Detection permis"
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.3
    NMS_THRESH = 0.3
    res = {}
    roi_file_name=[]
    if True:  
        cls_ind = 1 # CNI
        cls = CLASSES[cls_ind]
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        tot_info_cni = []
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            coef = 1.05
            pmax = im.shape[:2][::-1]
            for ind in xrange(4):
                if ind < 2:
                    bbox[ind] = bbox[ind] / coef
                else:
                    bbox[ind] = min(bbox[ind] * coef, pmax[ind - 2])
            print 'Saving recognized permis...'
            pts = [int(bx) for bx in bbox]
            filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename('output' + str(i) + image_name +'.png')
            filename = os.path.join(UPLOAD_FOLDER, filename_)
            cv2.imwrite(filename, im[pts[1]:pts[3], pts[0]:pts[2]])
            info_cni, roi_file_name=demo_parallel(net, filename)
            tot_info_cni.append(info_cni)
            #tot_info_cni.append(demo(net, filename))
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        return tot_info_cni, timer.total_time, roi_file_name
    im = im[:, :, (2, 1, 0)]
    return [(im, res, timer.total_time)], 0, roi_file_name  # equivalent to demo


"""demo2 is a complement for demo, in considering the multi-cni case 
    and if we should do faster-rcnn a second time"""

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        default=True,
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='axa')

    args = parser.parse_args()

    return args


def detect_permis_short(filename):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for classified permis...'
    return demo2(net, filename)

def calib_roi(im,bbx,cls):
    #txt, prob = clstm_ocr_carte_grise(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls=='lieu')
    txt, prob = clstm_ocr_permis(im[bbx[1]:bbx[3], bbx[0]:bbx[2]], cls)
    print txt, prob
    txt_temp,prob_temp="",0
    cv2.setNumThreads(0)
    h = np.size(im, 0)
    w = np.size(im, 1)
    if(prob<0.99):
        for i in range(1,4):    
            for j in range(1,3):
                #txt_temp,prob_temp=clstm_ocr_calib_carte_grise(im[bbx[1]-5*i*math.pow( -1, j):bbx[3]+5*i*math.pow( -1, j), bbx[0]-3*i*math.pow( -1, j):bbx[2]+3*i*math.pow( -1, j)], cls=='lieu')
                # if(cls=="numero" or cls=="marque" or cls=="type_mine"):
                #     txt_temp,prob_temp=clstm_ocr_calib_permis(im[bbx[1]-5*i*math.pow( -1, j):bbx[3]-5*i*math.pow( -1, j), bbx[0]-3*i*math.pow( -1, j):bbx[2]+3*i*math.pow( -1, j)], cls)
                #     print "calib_roi",  txt_temp, prob_temp, bbx[1]-5*i*math.pow( -1, j), bbx[3]-5*i*math.pow( -1, j)
                # else:  
                #     if (bbx[1]>15) and ( bbx[3] >15) and (bbx[2]>9) and (bbx[0]>9):
                #         txt_temp,prob_temp=clstm_ocr_calib_permis(im[bbx[1]-5*i*math.pow( -1, j):bbx[3]-5*i*math.pow( -1, j), bbx[0]-3*i*math.pow( -1, j):bbx[2]+3*i*math.pow( -1, j)], cls)
                if (bbx[1]>15) and ( bbx[3] >15) and (bbx[2]>9) and (bbx[0]>9):
                    txt_temp,prob_temp=clstm_ocr_calib_permis(im[bbx[1]-5*i*math.pow( -1, j):bbx[3]-5*i*math.pow( -1, j), bbx[0]-3*i*math.pow( -1, j):bbx[2]-3*i*math.pow( -1, j)], cls)
                    if(prob<prob_temp) and len(txt_temp)>=2 and checkdateregex(txt_temp,cls):
	                    txt=txt_temp
	                    prob=prob_temp

    if(prob<0.8 and cls=="prenom" and len(get_similar(txt,"prenom",0.7))>0 ):
        if abs(len(get_similar(txt,"prenom",0.7))-len(txt))<=1: 
            txt=txt+ ":" +get_similar(txt,"prenom",0.7)

    return txt, prob

def main():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    im_name = 'ID_FRA.jpg'
    # im_name = 'cni2.png'     
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for data/demo/{}'.format(im_name)
    demo2(net, im_name)

    plt.show()




if __name__ == '__main__':
    main()
