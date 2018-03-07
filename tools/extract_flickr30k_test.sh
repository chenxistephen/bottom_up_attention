# clear; python ./tools/generate_tsv.py --gpu 0 --split flickr30k --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --out features/flickr30k.tsv > experiments/logs/extract_flickr30k.log 2<&1

# clear; /usr/bin/python ./tools/generate_tsv.py --gpu 0,1,2,3 --split flickr30k_test --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --net data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --out features/flickr_all/flickr30k_test.tsv

clear; /usr/bin/python ./tools/generate_tsv.py --gpu 0,1,2,3 --split flickr30k_test --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --net data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --out features/roipool5_flickr/flickr30k_test.tsv