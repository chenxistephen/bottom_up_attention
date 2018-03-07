#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a 
   separate tsv file that can be merged later (e.g. by using merge_tsv function). 
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --caffe_net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014
###################################################
# Pytorch
import re
import caffe
import numpy as np
import skimage.io
from caffe.proto import caffe_pb2
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import resnet_frcnn
#from nets.resnet_v1 import resnet_v1
from collections import OrderedDict
from resnet_frcnn import resnet101
###################################################

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'roipool5', 'pool5', 'np_roipool5', 'np_pool5']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36. 
MIN_BOXES = 36 #10
MAX_BOXES = 36 #100

def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name.startswith('coco'): # COCO data
        from pycocotools.coco import COCO
        valAnnoFile = './data/coco/annotations/instances_val2014.json'
        trainAnnoFile = './data/coco/annotations/instances_train2014.json'
        if split_name == 'coco_dev':
            imgsetFile = './data/coco/ImageSets/devall_uniq_ids.txt'
        elif split_name == 'coco_train':
            imgsetFile = './data/coco/ImageSets/train_ids.txt'
        orgTrainList = [int(l.rstrip()) for l in open('./data/coco/ImageSets/train_sm_ids.txt')]
        orgTrainNum = len(orgTrainList)        
        valForTrainList = [int(l.rstrip()) for l in open('./data/coco/ImageSets/restval_ids.txt')]
        valForTrainNum = len(valForTrainList)
        print "orgTrainNum = {}, valForTrainNum = {}".format(orgTrainNum, valForTrainNum)
        imgids = [int(l.rstrip()) for l in open(imgsetFile)]
        imgids_train = [id for id in imgids if id in orgTrainList]
        print "imgids_train[:10] = {}".format(imgids_train[:10])
        imgids_val = [id for id in imgids if id not in orgTrainList]
        print "imgids_val[:10] = {}".format(imgids_val[:10])
        imgs_train = []
        imgs_val = []
        if len(imgids_train) > 0:
            cocoTrain=COCO(trainAnnoFile)
            imgs_train = cocoTrain.loadImgs(imgids_train)
        if len(imgids_val) > 0:
            cocoVal = COCO(valAnnoFile)
            imgs_val = cocoVal.loadImgs(imgids_val)

        print "appending image_ids"
        for image_id in imgids:
            if image_id in orgTrainList:
                idx = imgids_train.index(image_id)
                img = imgs_train[idx]
                filename = img['file_name']
                filepath = './data/coco/train2014/'
                filepath = osp.join(filepath, filename)
                #print filepath
                split.append((filepath,image_id))
            else: # val
                idx = imgids_val.index(image_id)
                img = imgs_val[idx]
                filename = img['file_name']
                filepath = './data/coco/val2014/'
                filepath = osp.join(filepath, filename)
                #print filepath
                split.append((filepath,image_id))
            if len(split) % 100 == 0:
                print len(split)
    elif split_name.startswith('flickr30k'):
        if split_name == 'flickr30k_val':
            imglist = [l.rstrip().split('\t') for l in open('./data/flickr30k/ImageSets/val.txt')]
        elif split_name == 'flickr30k_val':
            imglist = [l.rstrip().split('\t') for l in open('./data/flickr30k/ImageSets/val.txt')]
        elif split_name == 'flickr30k_train':
            imglist = [l.rstrip().split('\t') for l in open('./data/flickr30k/ImageSets/train.txt')]
        for img_name, image_id in imglist[:1]:
            filepath = os.path.join('./data/flickr30k/flickr30k_images/', img_name)
            print "filepath = {}, image_id = {}".format(filepath, image_id)
            split.append((filepath,image_id))
    elif split_name == 'coco_test2014':
      with open('/data/coco/annotations/image_info_test2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
      with open('/data/coco/annotations/image_info_test2015.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2015/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'genome':
      with open('/data/visualgenome/image_data.json') as f:
        for item in json.load(f):
          image_id = int(item['image_id'])
          filepath = os.path.join('/data/visualgenome/', item['url'].split('rak248/')[-1])
          split.append((filepath,image_id))      
    else:
      print 'Unknown split'
    return split

    
DEBUGFLAG = True
def get_detections_from_im(caffe_net, pt_net, im_file, image_id, conf_thresh=0.2):
    print "get_detections_from_im"
    print im_file
    print image_id
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(caffe_net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = caffe_net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = caffe_net.blobs['cls_prob'].data
    roipool5 = caffe_net.blobs['roipool5'].data
    pool5 = caffe_net.blobs['pool5_flat'].data
    
    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
    if DEBUGFLAG:
        roipool5 = roipool5[keep_boxes]
        pool5 = pool5[keep_boxes]
        features = Variable(torch.from_numpy(roipool5))
        #print("in features.size()= {}".format(features.size()))
        if 1:
            gt_roipool5_features = np.load('/home/chnxi/DoubleCrossAttVSE/roi_infeat_0.npy')
            d = np.linalg.norm(gt_roipool5_features - roipool5)
            print "d(gt_roipool5_features, caffe[roipool5= {}".format(d)
            features = Variable(torch.from_numpy(gt_roipool5_features))
        features = pt_net(features)      
        print("out features.size()= {}".format(features.size()))
        #features = features.view(batch_size, roiNum, -1)
        print("features.size() before fc = {}".format(features.size()))
        outfeat = features.data.numpy()
        print "outfeat: type = {}, shape = {}".format(type(outfeat), outfeat.shape)
        print "pool5: type = {}, shape = {}".format(type(pool5), pool5.shape)
        d = np.linalg.norm(outfeat-pool5)
        print "d = {}".format(d)
        print outfeat
        print pool5
        debugPath = '/home/chnxi/DoubleCrossAttVSE/debug_bottom/'
        np.save(debugPath + 'bn1.running_mean.npy', pt_net.layer4._modules['0'].bn1.running_mean.numpy())
        np.save(debugPath + 'bn1.running_var.npy', pt_net.layer4._modules['0'].bn1.running_var.numpy())
    else:
        # print "roipool5 dim = {}".format(roipool5[keep_boxes].shape)
        return {
            'image_id': image_id,
            'image_h': np.size(im, 0),
            'image_w': np.size(im, 1),
            'num_boxes' : len(keep_boxes),
            'boxes': base64.b64encode(cls_boxes[keep_boxes]),
            'roipool5': base64.b64encode(roipool5[keep_boxes]),
            'pool5': base64.b64encode(pool5[keep_boxes]), 
            'np_roipool5': roipool5[keep_boxes],
            'np_pool5': pool5[keep_boxes]
        }   


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_torch_cnn(saved_model):
    """Load a pretrained CNN and parallelize over GPUs
    """
    #saved_model = './data/torch/vg_resnet101.pth'
    net = resnet101()
    #net.create_architecture(21,tag='default', anchor_scales=[8, 16, 32])
    print "Loading model {}".format(saved_model)
    net.load_state_dict(torch.load(saved_model))
    net.eval()
    #net.cuda()
    #net = nn.DataParallel(net).cuda()
    return net
    
def generate_tsv(gpu_id, prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([int(image_id[1]) for image_id in image_ids])
    found_ids = set()
    outdir = osp.dirname(outfile)
    print "outdir = {}".format(outdir)
    if not osp.isdir(outdir):
        os.makedirs(outdir)
    #else:
    #    print "rm {}".format(outfile)
    #    os.system("rm {}".format(outfile))
    if os.path.exists(outfile):
        existCnt = 0
        print "Reading existing feature file: {}".format(outfile)
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                found_ids.add(int(item['image_id']))
                existCnt += 1
                if (existCnt + 1) % 100 == 0:
                    print existCnt+1
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print 'GPU {:d}: already completed {:d}'.format(gpu_id, len(image_ids))
    else:
        print 'GPU {:d}: missing {:d}/{:d}'.format(gpu_id, len(missing), len(image_ids))
    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        caffe_net = caffe.Net(prototxt, caffe.TEST, weights=weights)
        torch_model = '/home/chnxi/bottom-up-attention/data/torch/vg_resnet101.pth' #'./data/torch/vg_resnet101.pth'
        pt_net = get_torch_cnn(torch_model)
        out_roi_file = osp.join(outdir, 'tmp_roipool5.npy')
        out_pool_file = osp.join(outdir, 'tmp_pool5.npy')
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
            _t = {'misc' : Timer()}
            count = 0
            roi_data = []
            pool_data = []
            for im_file,image_id in image_ids:
                print im_file
                print image_id
                if True: #int(image_id) in missing:
                    _t['misc'].tic()
                    item = get_detections_from_im(caffe_net, pt_net, im_file, image_id)
                    if not DEBUGFLAG:
                        roi_data.append(item['np_roipool5'])
                        pool_data.append(item['np_pool5'])
                        writer.writerow(item)
                    _t['misc'].toc()
                    print "{}/{}".format(count+1, len(image_ids))
                    if (count % 100) == 0:
                        print 'GPU {:d}: {:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                              .format(gpu_id, count+1, len(missing), _t['misc'].average_time, 
                              _t['misc'].average_time*(len(missing)-count)/3600)
                    count += 1
        if not DEBUGFLAG:
            np.save(out_roi_file, np.stack(roi_data, axis=0))
            np.save(out_pool_file, np.stack(pool_data, axis=0))

                    


def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]

    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        
        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                      writer.writerow(item)
                    except Exception as e:
                      print e                           

    
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.data_split)
    # random.seed(10)
    # random.shuffle(image_ids)
    # Split image ids between gpus
    #image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    gpu_image_ids = []
    splen = np.ceil(len(image_ids)/float(len(gpus))).astype(int)
    print "Split {} images for each of {} GPUS".format(splen, len(gpus))
    for i in range(len(gpus)):
        start = i * splen
        end = min( (i+1) * splen, len(image_ids))
        print range(start,end)
        gpu_image_ids.append(image_ids[start:end])
    # print gpu_image_ids
    #sys.exit()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []    
    
    for i,gpu_id in enumerate(gpus):
        outfile = '%s.%d' % (args.outfile, gpu_id)
        p = Process(target=generate_tsv,
                    args=(gpu_id, args.prototxt, args.caffemodel, gpu_image_ids[i], outfile))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()            
                  
