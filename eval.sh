#!/bin/sh

#python eval.py  --test -t iou -m  yolo  --mode precision  -w weightscomplete/yolov3-tiny_best_288.pt
python eval.py  --test -t iou -m  yolo  --mode realtime  -v -w weightscomplete/yolov3-tiny_best_288.pt
