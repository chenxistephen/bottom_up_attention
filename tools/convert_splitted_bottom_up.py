#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import json
from collections import defaultdict


target_split = sys.argv[3] if len(sys.argv) > 3 else 'train'
sampleNum = int(sys.argv[4]) if len(sys.argv) > 4 else 256
json_data=open("/media/data/kualee/vsepp/data/f30k/dataset_flickr30k.json").read()

data = json.loads(json_data)['images']
meta = {}

for d in data:
    imgid = d['imgid']
    split = d['split']
    meta[imgid] = split


csv.field_size_limit(sys.maxsize)
   
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = sys.argv[1] #'/home/chnxi/bottom-up-attention/features/roipool5_flickr/flickr30k_{}_merged.tsv'.format(target_split)
outFileName = sys.argv[2] #'/home/chnxi/bottom-up-attention/features/roipool5_npy/{}_ims.npy'.format(target_split)

if __name__ == '__main__':
    print "infile = {}".format(infile)
    print "outFleName = {}".format(outFileName)
    print "target_split = {}".format(target_split)
    print "sampleNum = {}".format(sampleNum)

    # Verify we can read a tsv
    data = {}
    #ids = []
    # test = {}
    # val = {}
    count = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            count += 1
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            # print(item['image_id'])
            for field in ['boxes', 'features']:
                # print(field, len(item[field]))
                buf = base64.decodestring(item[field])
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item['num_boxes'],-1))
            #in_data[item['image_id']] = item
            # if split == 'train':
            #     train[item['image_id']] = item['features']
            # elif split == 'test':
            #     test[item['image_id']] = item['features']
            # else:
            #     val[item['image_id']] = item['features']
            if meta[item['image_id']] != target_split:
                print(item['image_id'], meta[item['image_id']])
                #return
            data[item['image_id']] = item['features']
            #ids.append(item['image_id'])
            #in_data[item['image_id']] = item['features']
            print "{}/{}".format(count, sampleNum)
            if count == sampleNum:
                break
    data_imgid = sorted(data.keys())
    print(len(data))
    #print(data_imgid == ids)
    
    if target_split == 'train':
        out = np.stack([data[i] for i in data_imgid], axis=0)
    else:
        out = []
        for imid in data_imgid:
            out.append(data[imid])
            out.append(data[imid])
            out.append(data[imid])
            out.append(data[imid])
            out.append(data[imid])
        out = np.stack(out, axis=0)
    np.save(outFileName, out)
    # test_imgid = sorted(test.keys())
    # val_imgid = sorted(val.keys())
    # train_out = np.stack([train[i] for i in train_imgid], axis=0)
    # if 
    # train_out = np.stack([data[i] for i in train_imgid], axis=0)
    # test_out = []
    # for i in test_imgid:
    #     test_out.append(test[i])
    #     test_out.append(test[i])
    #     test_out.append(test[i])
    #     test_out.append(test[i])
    #     test_out.append(test[i])
    # val_out = []
    # for i in val_imgid:
    #     val_out.append(val[i])
    #     val_out.append(val[i])
    #     val_out.append(val[i])
    #     val_out.append(val[i])
    #     val_out.append(val[i])
    # test_out = np.stack(test_out, axis=0)
    # val_out = np.stack(val_out, axis=0)
    # print(train_out.shape, test_out.shape, val_out.shape)
    # np.save('/media/data/kualee/flickr_bottom_up_feature/train_ims.npy', train_out)
    # np.save('/media/data/kualee/flickr_bottom_up_feature/test_ims.npy', test_out)
    # np.save('/media/data/kualee/flickr_bottom_up_feature/dev_ims.npy', val_out)

            #print(item['features'].shape)
            #break
    #print in_data['features']


