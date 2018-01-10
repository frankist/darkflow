#!/bin/bash

cd ..
./flow --imgdir ../test_voc/JPEGImages --model data/yolo_example.cfg --load -1
./flow --imgdir ../test_voc/JPEGImages --model data/yolo_example.cfg --load -1 --json
#./flow --imgdir ../RFVOCFormat2/JPEGImages --model data/yolo_new_anchors.cfg --load -1
#./flow --imgdir ../RFVOCFormat2/JPEGImages --model data/yolo_new_anchors.cfg --load -1 --json
