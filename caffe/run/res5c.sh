#!/bin/bash

cd python;
python extract_image_features.py --modelDir=../models/ResNet --gpu --outTsvFile=outTsvFile.tsv --device_id=0
