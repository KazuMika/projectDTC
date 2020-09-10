# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.serialization import load_lua
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from data import VOC_CLASSES as labels
from trash import Trash
import time
from random import randint
from fpsrate import FpsWithTick
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--test', default=True, help='sum the integers (default: find the max)')
parser.add_argument('--frame_rate','-f', default=18.0,type=float, help='sum the integers (default: find the max)')

args = parser.parse_args()
IOU_TH=0

SSD_TH=0.6
COLOR = ((255, 130, 1),(0, 255, 255),(0, 252, 124),
        (51, 102, 255),(91, 10, 219),(239, 79, 117))


def bSort(a):
    for i in range(len(a)):
        for j in range(len(a)-1, i, -1):
            if a[j][3] > a[j-1][3]:
                a[j], a[j-1] = a[j-1], a[j]
    return a


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def bb_intersection_over_union(boxA, boxB):
#    print("BoxA",boxA)
 #   print("BoxB",boxB)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = (xB - xA + 1) * (yB - yA + 1)
    xinter = (xB - xA + 1)
    yinter = (yB - yA + 1)
    if xinter <= 0 or yinter <= 0:
        iou = 0
        return iou
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if iou < 0 or iou > 1:
        iou = 0
  #  print(iou)
    return iou


def detect_image(frame,net):
    x = cv2.resize(frame, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)

    x = x.astype(np.float32)

    x = x[:, :, ::-1].copy()

    x = torch.from_numpy(x).permute(2, 0, 1)
    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y = net(xx)
    top_k=10

    detections = y.data
    scale = torch.Tensor([frame.shape[1::-1], frame.shape[1::-1]])
    cords = []
    texts = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= SSD_TH:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            label_name = labels[i-1]
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            cords.append(pt)
            texts.append(display_txt)
            j+=1

    return cords, texts

def match_trash(cord,trash,_ies):
    if len(trash) == 0:
        return False
    ious =[]
    trash_ids = []
    for i in trash:
        if i.id in _ies:
            continue
        iou = bb_intersection_over_union(i.cords,cord)
        ious.append(iou)
        trash_ids.append(i.id)

    ious = np.array(ious)
    ious2 = ious > IOU_TH
    trash_ids = np.array(trash_ids)

    if ious.shape[0] == 0:
        return False
    else:
        if True in ious2:
            return int(trash_ids[ious.argmax()])
        else:
            return False


def make_avi(path_to_avi,net, frame_rate):

    cap = cv2.VideoCapture(path_to_avi)
    #fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    baseavi = os.path.basename(path_to_avi).replace(".avi",".mp4")
    path_to_avi = path_to_avi.replace(".avi",".mp4").replace("../../","").replace("/","_")
    print(path_to_avi)
    video = cv2.VideoWriter("./test/" + path_to_avi, fourcc, frame_rate, (int(cap.get(3)),int(cap.get(4))))

    cnt_down = 0

    height = cap.get(4)
    line_down   = int(9*(height/10))
    up_limit =   int(0)
    down_limit = int(height)

    font = cv2.FONT_HERSHEY_DUPLEX
    trashs = []
    max_age = 2
    t_id = 0
    frame_count =0
    fps_count= 0
    fpsWithTick = FpsWithTick()
    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            break

        cords, texts = detect_image(rgb_image,net)
        cords = bSort(cords)
        detected_trash_ids = []


        for cord,text in zip(cords,texts):
        
            cord = cord.astype("int64")
            x,y,w,h =cord[0],cord[1], cord[2]-cord[0],cord[3]-cord[1]
            _center = np.array([int(x + (w/2)),int(y +(h*7/8))])
            new = True
#            if _center[1] in range(up_limit,down_limit):
            trash_i = match_trash(cord,trashs,detected_trash_ids)
            if isinstance(trash_i, int) :
                for i in trashs:
                    if i.id==trash_i:
                        new = False
                        detected_trash_ids.append(i.id)
                        i.updateCoords(cord,_center) 
                        if i.going_DOWN(line_down) == True:
                            if i.state == True:
                                cnt_down += 1;
                                i.state = False
                                i.done = True
                                print ("ID:",i.id,'crossed')
            if new == True and _center[1] < line_down:
                color_trash = COLOR[2]
                t = Trash(t_id,cord, _center,max_age,color_trash)
                trashs.append(t)
                detected_trash_ids.append(t.id)
                t_id += 1     

            #cv2.circle(frame,(_center[0],_center[1]), 3, (0,0,255), -1)
            #cv2.rectangle(frame,(cord[0],cord[1]),(cord[2],cord[3]),COLOR[2],2)            


        for i in trashs:
            i.age += 1
            if i.age > max_age:
                i.done = True

            cv2.circle(frame,(i.center[0],i.center[1]), 3, (0,0,126), -1)
            cv2.rectangle(frame,(i.cords[0],i.cords[1]),(i.cords[2],i.cords[3]),i.color,2)            

            cv2.rectangle(frame, (i.cords[0], i.cords[1] - 20),
                                  (i.cords[0] + 170, i.cords[1]), i.color, thickness=2)
            cv2.rectangle(frame, (i.cords[0], i.cords[1] - 20),
                                  (i.cords[0] + 170, i.cords[1]),i.color, -1)
            cv2.putText(frame, str(i.id) + " " +str(i.age)+" " + text, (i.cords[0], i.cords[1] - 5),font,0.6, (0, 0, 0), 1,cv2.LINE_AA)
            if i.center[1] > line_down:
                i.done = True
            if i.done:
                index = trashs.index(i)
                trashs.pop(index)


        str_down = 'COUNT:'+ str(cnt_down)
        cv2.line(frame,(0,line_down),(int(cap.get(3)),line_down),(255,0,0),2)
        cv2.putText(frame, str_down ,(10,70),font,2.5,(0,0,0),10,cv2.LINE_AA)
        cv2.putText(frame, str_down ,(10,70),font,2.5,(255,255,255),8,cv2.LINE_AA)
      #  video.write(frame)
        fps1= fpsWithTick.get()
        fps_count += fps1
        frame_count += 1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    print("Avarage fps : {0:.2f}".format(fps_count / (frame_count-1)))
    cap.release()
    video.release()
    cv2.destroyAllWindows()


def main():
    if not os.path.exists("./test"):
        os.mkdir("./test")
    net = build_ssd('test', 300, 2)
    net.load_weights('./weights/ssd300_0712_160000.pth')
    if not os.path.exists("./test"):
        os.mkdir("./test")
    if args.test==True:
        avis = "testgomi2.avi"
        make_avi(avis,net, args.frame_rate)
    else:
        avis = [avi for avi in find_all_files("../../back_kanen_1_8_data/") if ".avi" in avi]
        avis.sort()
        for avi in avis:
            path2 = avi.replace(".avi",".mp4").replace("../../","").replace("/","_")
            path2 = "./test/" +path2
            if os.path.exists(path2):
                continue
            make_avi(avi,net,args.frame_rate)



if __name__ == "__main__":
    main()

