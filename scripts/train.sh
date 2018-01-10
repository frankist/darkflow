#!/bin/bash

cd ..
./flow --model data/yolo_example.cfg --train --dataset "../test_voc/JPEGImages/" --annotation "../test_voc/Annotations/" --epoch 10000000 #--load -1
#./flow --model data/yolo_example.cfg --train --dataset "../RFVOCFormat2/JPEGImages/" --annotation "../RFVOCFormat2/Annotations/" --epoch 10000000 #--load -1
#./flow --model data/yolo_new_anchors.cfg --train --dataset "../test_voc/JPEGImages/" --annotation "../test_voc/Annotations/" --epoch 10000000 #--load -1 
#./flow --model data/yolo_new_anchors.cfg --train --dataset "../RFVOCFormat2/JPEGImages/" --annotation "../RFVOCFormat2/Annotations/" --epoch 10000000 #--load -1 
