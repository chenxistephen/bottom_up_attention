#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import os.path as osp

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
csv.field_size_limit(sys.maxsize)

imgsetFolder = '/home/chnxi/data/flickr30k/ImageSets'
splits = ['train','val','test']
feature_imglist_file = '/home/chnxi/data/flickr30k/imglist.txt'
imglist = [l.rstrip().split('.')[0] for l in open(feature_imglist_file,'r').readlines()]
# name_ids_file = '/home/chnxi/data/flickr30k/flickr_imgname_ids.tsv'

if __name__ == '__main__':

    infile = sys.argv[1] if len(sys.argv) > 1 else './features/bottom_up/flickr30k_merged.tsv'
    reader = csv.DictReader(open(infile, "r+b"), delimiter='\t', fieldnames = FIELDNAMES)
    print "generating merged_names"
    merged_names = [imglist[int(item['image_id'])] for item in reader]
    print merged_names[0]
    print "reading lines"
    lines = open(infile,'r').readlines()
    
    for set in splits:
        outfile = osp.splitext(infile)[0] + '_{}_sorted.tsv'.format(set)
        print outfile
        name_ids_file = osp.join(imgsetFolder, set + '.txt')
        sorted_names = [l.split('\t')[0].split('.')[0] for l in open(name_ids_file,'r').readlines()]
        print sorted_names[0]        
        with open(outfile,'wb') as fout:
            for name in sorted_names:
                imgidx = merged_names.index(name)
                print "{}:{}".format(name, imgidx)
                line = lines[imgidx]
                fout.write(line)


