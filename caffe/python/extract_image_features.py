#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import matplotlib.pyplot as plt
import caffe
import os.path as osp
from glob import glob
from timer import Timer

import base64
import cv2
import csv
from multiprocessing import Process
import random
import json

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']



imgsetFile = '../../data/flickr30k/imglist.txt'

imgPath = '../../data/flickr30k/flickr30k_images/'

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)


    input_scale = 1.0 #0.017 # 1.0
    raw_scale = 255.0 # [1.0, 255.0]
    PIXEL_MEANS = np.array([[[103.94/255.0, 116.78/255.0, 123.68/255.0]]])
    vis = False
    saveVis = True
    runEval = True
    
    imglist = [l.rstrip() for l in open(imgsetFile)]



    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "--modelDir",
        #default='../../trained_models/VisualIntent/caffenet/softmax_50k/',
        help="Dir of the trained model"
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--cpu",
        default=False,
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0, #255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--input_file",
        default=imglist,
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "--output_dir",
        default='../outputs',
        help="Output npy filename."
    )

    parser.add_argument(
        "--outTsvFile",
        default=None,
        help="Output npy filename."
    )
    # Optional arguments.
    # parser.add_argument(
        # "--model_def",
        # default=os.path.join(pycaffe_dir,
                # modelDir, "deploy.prototxt"),
        # help="Model definition file."
    # )
    # parser.add_argument(
        # "--pretrained_model",
        # default=os.path.join(pycaffe_dir,
                # modelDir, "caffenet_softmax_iter_50000.caffemodel"),
        # help="Trained model weights file."
    # )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='512,512',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default='', #os.path.join(pycaffe_dir,'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=input_scale,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=raw_scale, #255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()
    deploy_prototxt = glob(osp.join(args.modelDir, '*.prototxt'))[0] #osp.join(args.modelDir, 'deploy.prototxt')
    trained_model = glob(osp.join(args.modelDir, '*.caffemodel'))[0]
    
    # PIXEL_MEANS = np.array([[[128.223/255.0,135.409/255.0,142.853/255.0]]])
    # PIXEL_MEANS = np.array([[[103.94, 116.78, 123.68]]])
    
    
    print PIXEL_MEANS
    T = Timer()
    output_dir = args.output_dir
    visDir = osp.join(output_dir, 'visualization')
    resDir = osp.join(output_dir, 'pred_results')
    modelName = osp.splitext(osp.basename(trained_model))[0]
    print modelName
    if "mobilenet" in modelName or "mob" in modelName: # or "googlenet" in args.modelDir:
        print "change args.input_scale to 0.017!========================"
        args.input_scale = 0.017 # 1.0
    visFolder = osp.join(visDir, modelName)
    outresFolder = osp.join(resDir, modelName)
    outTsvFile = osp.join(outresFolder, modelName)
    if vis and saveVis and not osp.isdir(visFolder):
        os.makedirs(visFolder)
    if not osp.isdir(outresFolder):
        os.makedirs(outresFolder)

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]
    print mean
    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(args.device_id)
        print("=============================== GPU mode, device_id: {} ===============================".format(args.device_id))
    elif args.cpu:
        caffe.set_mode_cpu()
        print("=============================== CPU mode ===============================")
        

    # Make classifier.
    classifier = caffe.Classifier(deploy_prototxt, trained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)
    print "device_id = {}".format(args.device_id)
    print "deploy_prototxt = {}".format(deploy_prototxt)
    print "trained_model = {}".format(trained_model)
    print "args.input_scale = {}".format(args.input_scale)
    print "args.raw_scale = {}".format(args.raw_scale)
    print "args.gpu: {}".format(args.gpu)
    print "len(imglist) = {}=============================".format(len(imglist))



    # Classify.
    start = time.time()
    # predictions = classifier.predict(inputs, not args.center_only)
    # print predictions
    conf_thrsh = 0.1 #01
    overallAccuracy = 0
    OUT_WIDTH = 14
    OUT_HEIGHT = 14
    DIM = 2048
    all_scores = []
    all_img_names = []
    with open(outTsvFile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        _t = {'misc' : Timer()}
        count = 0
        for ix, im_f in enumerate(imglist):
            imgFile = imgPath + im_f
            print imgFile
            all_img_names.append(im_f)
            img = caffe.io.load_image(imgFile)
            print img.shape
            inputs = [img - PIXEL_MEANS]
            T.tic()
            # Axis order will become: (batch elem, channel, height, width)
            predictions = classifier.predict(inputs, not args.center_only)
            T.toc()
            print ("{}:{:.3f}s".format(ix, T.average_time))
            imgName = osp.basename(im_f)
            features = predictions[0]
            print features.shape
            # features = np.reshape(features, (DIM, OUT_HEIGHT, OUT_WIDTH))
            # print features.shape
            # channel_swap = (1,2,0) # HEIGHT, WIDTH, DIM
            # features = np.transpose(features, (channel_swap))
            # print features.shape
            row = {'image_id': ix,
                    'image_h': img.shape[0],
                    'image_w': img.shape[1],
                    'num_boxes' : OUT_HEIGHT * OUT_WIDTH,
                    'boxes': base64.b64encode(np.arange(OUT_HEIGHT*OUT_WIDTH)),
                    'features': base64.b64encode(features)
                  }
            writer.writerow(row)
        # print scores

    print("Done in %.2f s." % (time.time() - start))
    overallAccuracy /= len(imglist)
    print "Overall Multilabel Accuracy = {:.3f}".format(overallAccuracy)
    



if __name__ == '__main__':
    main(sys.argv)
