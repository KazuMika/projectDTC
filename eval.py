# -*- coding: utf-8 -*-
import os
import torch.backends.cudnn as cudnn
import argparse
from counter import Counter
import time

cudnn.benchmark = True


def main():
    '''
    Complete

    precision_yolo_sort
    precision_yolo_iou
    fps_yolo_sort
    fps_yolo_iou
    visualization_yolo_sort
    visualization_yolo_iou
    count_on_jetson_yolo_sort
    count_on_jetson_yolo_iou
    precision_ssd_sort
    precision_sdd_iou
    fps_ssd_sort
    fps_ssd_iou
    visualization_ssd_sort
    visualization_ssd_iou
    count_on_jetson_ssd_sort
    count_on_jetson_ssd_iou
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',
                        action='store_true',
                        help='only testgom.mp4')
    parser.add_argument('--cfg',
                        type=str,
                        default='cfg/yolov3-tiny.cfg',
                        help='cfg file path if yolo')
    parser.add_argument('--save_dir_path',
                        type=str,
                        default='/home/quantan/DTCEvaluation/yolov3_dtceval/results/',
                        help='cfg file path if yolo')
    parser.add_argument('--weights',
                        '-w',
                        type=str,
                        default='./weightscomplete/yolov3-tiny_best_288.pt',
                        help='path to weights file')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.40,
                        help='object confidence threshold')
    parser.add_argument('--nms-thres',
                        type=float,
                        default=0.45,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--mode',
                        type=str,
                        default='precision',
                        help='precision or \
                                visualization or jetson \
                                realtime')
    parser.add_argument('--tracking_alg',
                        '-t',
                        type=str,
                        default='iou',
                        help="iou or sort")
    parser.add_argument('--img-size',
                        type=int,
                        default=32 * 9,
                        help='size of image dimension')
    parser.add_argument('--model',
                        '-m',
                        type=str,
                        default='ssd',
                        help='ssd or yolov3')
    parser.add_argument('--video', '-v',
                        action='store_true')
    parser.add_argument('--fps_eval', '-f',
                        action='store_true')

    opt = parser.parse_args()
    print(opt)

    counter = Counter(opt)
    counter.execution()


if __name__ == "__main__":

    t = time.time()
    main()
    t = time.time() - t
    print('end_time:{0:0.3f} seconds'.format(t))
