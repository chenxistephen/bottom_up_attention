clear; python ./tools/generate_resnet_feature_tsv.py --gpu 0,1,2,3 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def ./caffe/models/ResNet/ResNet-152-deploy.prototxt --out ./features/Res152_flickr30k.tsv --net ./caffe/models/ResNet/ResNet-152-model.caffemodel --split flickr30k

