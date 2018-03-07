import sys
import os.path as osp
import json

datasetFile = '/media/data/kualee/vsepp/data/f30k/dataset_flickr30k.json'

outFolder = '/home/chnxi/data/flickr30k/ImageSets'
splits = ['train','val','test']


d = json.load(open(datasetFile,'r'))
d = d['images']

for set in splits:
    outFileName = osp.join(outFolder, set + '.txt')
    name_ids = [(img['filename'], img['imgid']) for img in d if img['split'] == set]
    print name_ids[0]
    with open(outFileName,'wb') as fout:
        for k,v in name_ids:
            fout.write('{}\t{}\n'.format(k,v))