# import _init_paths
# from fast_rcnn.config import cfg, cfg_from_file
# from fast_rcnn.test import im_extract_features,_get_blobs
# from fast_rcnn.nms_wrapper import nms
# from utils.timer import Timer

# import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp
import base64
import numpy as np

import csv
import json
from glob import glob

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

csv.field_size_limit(sys.maxsize)


if __name__ == '__main__':
    testParse = True
    prefix = sys.argv[1] if len(sys.argv) > 1 else './features/bottom_up/flickr30k.tsv'
    outfile = osp.splitext(prefix)[0] + '_merged.tsv'
    tsvlist = [prefix + '.{}'.format(i) for i in range(4)]
    print tsvlist

    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)   
        
        for infile in tsvlist:
            print infile
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                        # if testParse:
                            # for field in ['boxes', 'features']:
                                # tmp = np.frombuffer(base64.decodestring(item[field]), dtype=np.float32).reshape((item['num_boxes'],-1))
                        writer.writerow(item)
                    except Exception as e:
                      print e
