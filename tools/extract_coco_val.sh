clear; /usr/bin/python ./tools/generate_tsv.py --gpu 2,3 --split coco_dev --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --net data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel --out features/coco_roipool/coco_val.tsv