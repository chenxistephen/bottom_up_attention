#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'roipool5','pool5']
# infile = '/data/coco/tsv/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv'



if __name__ == '__main__':
    infile = sys.argv[1]
    # Verify we can read a tsv
    in_data = {}
    poolh = 14
    poolw = 14
    dim = 1024
    count = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            #print item['image_id']
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            
            for field in ['boxes', 'roipool5']:
                if field == 'boxes':
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'],-1))
                elif field == 'roipool5':
                    item[field] = np.frombuffer(base64.decodestring(item[field]), 
                          dtype=np.float32).reshape((item['num_boxes'], dim, poolh, poolw))
                if (count + 1) % 100 == 0:
                    print "{}: item['roipool5'] dimension = {}".format(count+1, item['roipool5'].shape)
            in_data[item['image_id']] = item
    #print in_data
    print len(in_data)


