# -*- coding: utf-8 --
import cv2
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn
import csv
from sort import Sort
from models import Darknet
from utils import torch_utils
from utils.utils import non_max_suppression, scale_coords
from utils.datasets import letterbox
from iou_tracking import Iou_Tracker
from utils.count_utils import find_all_files
import datetime
from fpsrate import FpsWithTick
import random
import threading
from ssd_src.ssd import build_ssd
from collections import deque
import math

cudnn.benchmark = True


class Counter(object):
    def __init__(self, args):
        self.fpsWithTick = FpsWithTick()
        self.frame_count = 0
        self.fps_count = 0
        self.flag_of_recall_q = ''
        self.flag_of_detection_q = ''
        self.p = 0
        self.q = deque()
        self.i = 0
        self.flag_of_realtime = True
        self.recallq = deque()
        self.flag_of_detection_q = False
        self.flag_of_recall_q = True
        self.args = args
        self.path = self.args.save_dir_path
        self.conf_thres = self.args.conf_thres
        self.nms_thres = self.args.nms_thres
        self.img_size = self.args.img_size
        self.device = torch_utils.select_device()
        self.y_pred_list = []
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.max_age = 2
        self.ssd_th = 0.4
        self.frame_rate = 20
        self.mode = self.args.mode
        self.tracking_alg = self.args.tracking_alg
        self.model = args.model
        self.fps_eval = self.args.fps_eval
        self.video = self.args.video
        self.l = 10  # the lower bound of processing rate  by koyo
        self.w = 0  # the window size of frames in which objects are not detected

        if self.model == 'yolo':
            print("yolo")
            self.net = Darknet(self.args.cfg, self.args.img_size)
            self.net.load_state_dict(torch.load(
                self.args.weights, map_location='cpu')['model'])
            self.net.to(self.device).eval()
        elif self.model == 'ssd':
            print("ssd")
            if torch.cuda.is_available():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.net = build_ssd('test', 300, 2)
            self.net.load_weights('./weightscomplete/v2r.pth')
            self.net.eval()

        self.prepare()

        f = open(os.path.join(self.save_root_dir, 'y_pred.csv'), "w")
        self.writer = csv.writer(f, lineterminator='\n')

    def prepare(self):
        if self.mode == 'precision':
            self.save_root_dir = self.path
            self.image_dir = os.path.join(self.save_root_dir, 'images')
            self.movie_dir = os.path.join(self.save_root_dir, 'movies')
            if not os.path.exists(os.path.join(self.image_dir)):
                os.makedirs(os.path.join(self.image_dir))
            if not os.path.exists(os.path.join(self.movie_dir)):
                os.makedirs(os.path.join(self.movie_dir))
            if self.args.test:
                self.movies = ['./testgomi2.mp4']
            else:
                self.movies = [movie for movie in find_all_files(
                    '/home/quantan/back_kanen_1_8_data_mp4/') if '.mp4' in movie]
                self.movies.sort()
        elif self.mode == 'visualization':
            self.save_root_dir = self.path
            self.image_dir = os.path.join(self.save_root_dir, 'images')
            self.movie_dir = os.path.join(self.save_root_dir, 'movies')
            self.gps_datelist = []
            self.gps_list = []
            self.gps_dir = 'visualize_' + self.model
            self.gps_image_dir = 'gps_count_images'
            self.gps_location_dir = 'gps_locations_data'
            if not os.path.exists(os.path.join(self.save_root_dir,
                                               self.gps_dir, self.gps_image_dir)):
                os.makedirs(os.path.join(self.save_root_dir,
                                         self.gps_dir, self.gps_image_dir))
            if not os.path.exists(os.path.join(self.save_root_dir,
                                               self.gps_dir, self.gps_location_dir)):
                os.makedirs(os.path.join(self.save_root_dir,
                                         self.gps_dir, self.gps_location_dir))
            self.movies = [movie for movie in find_all_files(
                '/home/quantan/dataset/datavisual') if "CH1.264" in movie]
            self.movies.sort()

        elif self.mode == 'jetson':
            self.save_root_dir = '/mnt/hdd1/'
            self.save_image_dir = os.path.join(self.save_root_dir, 'image_results/')
            self.save_movie_dir = os.path.join(self.save_root_dir, 'movie_results/')
            if not os.path.exists(self.save_image_dir):
                os.makedirs(self.save_image_dir)
            if not os.path.exists(self.save_movie_dir):
                os.makedirs(self.save_movie_dir)

        elif self.mode == 'realtime':
            self.save_root_dir = self.path
            self.image_dir = os.path.join(self.save_root_dir, 'images')
            self.movie_dir = os.path.join(self.save_root_dir, 'movies')
            if not os.path.exists(os.path.join(self.image_dir)):
                os.makedirs(os.path.join(self.image_dir))
            if not os.path.exists(os.path.join(self.movie_dir)):
                os.makedirs(os.path.join(self.movie_dir))
            if self.args.test:
                self.movies = ['./testgomi2.mp4']
            else:
                self.movies = [movie for movie in find_all_files(
                    '/home/quantan/back_kanen_1_8_data_mp4/') if '.mp4' in movie]
                self.movies.sort()
        else:
            pass

    def execution(self):
        if self.mode == 'precision':
            for movie in self.movies:
                print(movie)
                self.evalate_precision(movie)

            self.writer.writerows(self.y_pred_list)

        elif self.mode == 'visualization':
            for movie in self.movies:
                print(movie)
                datevs = movie.split('/')[5]

                if len(self.gps_datelist) == 0:
                    self.gps_datelist.append(datevs)
                    gps_log = open(os.path.join(
                        self.save_root_dir, self.gps_dir, self.gps_location_dir, datevs+'.csv'), 'w')
                    writer2 = csv.writer(gps_log, lineterminator='\n')
                    self.gps_list = []
                elif datevs not in self.gps_datelist:
                    self.gps_datelist.append(datevs)
                    writer2.writerows(self.gps_list)
                    gps_log.close()
                    gps_log = open(os.path.join(
                        self.save_root_dir, self.gps_dir, self.gps_location_dir, datevs+'.csv'), 'w')
                    self.gps_list = []
                    writer2 = csv.writer(gps_log, lineterminator='\n')

                self.visualization(movie)

            writer2.writerows(self.gps_list)
            gps_log.close()
            self.writer.writerows(self.y_pred_list)
        elif self.mode == 'jetson':
            self.counting_on_jetson()
        elif self.mode == 'realtime':
            for movie in self.movies:
                print(movie)
                self.realtime_detection(movie)

            self.writer.writerows(self.y_pred_list)

        else:
            pass

    def evalate_precision(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie).replace('.mp4', '')
        movie_id = basename[0:4]

        save_movie_path = os.path.join(self.movie_dir, basename+'.mp4')
        if self.video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(save_movie_path, fourcc,
                                    self.frame_rate, (int(cap.get(3)), int(cap.get(4))))
        height = cap.get(4)
        line_down = int(9*(height/10))

        if self.tracking_alg == 'sort':
            tracker = Sort(1, self.max_age, line_down, movie_id,
                           self.image_dir, '', basename)
        else:
            tracker = Iou_Tracker(
                line_down, self.image_dir, movie_id, self.max_age, '', basename)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'ssd':
                cords = np.array(self.detect_image(rgb_image))
            else:
                cords = self.detect_image(rgb_image)

            tracker.update(cords, frame, fps_eval=self.fps_eval)

            if self.video:
                video.write(frame)

        if self.fps_eval:
            print("Avarage fps : {0:.2f}".format(
                tracker.fps_count / tracker.frame_count))

        self.y_pred_list.append((movie_id, tracker.cnt_down))
        cap.release()
        cv2.destroyAllWindows()

        if self.video:
            video.release()

    def visualization(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie)
        movie_date = path_to_movie.split('/')[-3]

        save_image_dir = os.path.join(
            self.save_root_dir, self.gps_dir, self.gps_image_dir)
        movie_id = basename[0:4]

        height = cap.get(4)
        line_down = int(9*(height/10))

        gps_path = path_to_movie.replace('H264/CH1.264', 'SNS/Sns.txt')
        if not os.path.exists(gps_path):
            return None

        f2 = open(gps_path, 'r')
        gpss = [gps.strip() for gps in f2.readlines()]
        gps_count = 0
        if self.tracking_alg == 'sort':
            tracker = Sort(1, 3, line_down, movie_id,
                           save_image_dir, movie_date)
        else:
            tracker = Iou_Tracker(line_down, save_image_dir,
                                  movie_id, 2, movie_date)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'sort':
                cords = np.array(self.detect_image(frame))
            else:

                cords = self.detect_image(rgb_image)
            tracker.update(cords, frame, gpss=gpss, gps_count=gps_count,
                           visualize=True, gps_list=self.gps_list)
            gps_count += 1

        cap.release()
        # video.release()
        cv2.destroyAllWindows()
        self.y_pred_list.append((movie_id, tracker.cnt_down))

    def counting_on_jetson(self):
        cap = cv2.VideoCapture('/home/quantan/DTCEvaluation/yolov3_dtceval/testgomi2.mp4')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # cap = cv2.VideoCapture('testgomi2.mp4')
        time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        save_movie_dir = os.path.join(self.save_movie_dir, (time_stamp+'.avi'))
        if not os.path.exists(self.save_image_dir):
            os.mkdir(self.save_image_dir)

        # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(save_movie_dir, fourcc,
                                20, (int(cap.get(3)), int(cap.get(4))))

        prediction = []
        prediction2 = []

        height = cap.get(4)
        line_down = int(9*(height/10))

        frame_count = 0
        fps_count = 0
        fpsWithTick = FpsWithTick()
        count = 0
        if self.tracking_alg == 'sort':
            tracker = Sort(1, 3, line_down,
                           save_image_dir=save_movie_dir)
        else:
            tracker = Iou_Tracker(line_down, save_image_dir=self.save_image_dir,
                                  save_movie_dir=save_movie_dir)

        while(cap.isOpened()):
            time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
            count += 1
            ret, frame = cap.read()
            if ret:
                frame2 = frame.copy()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                break

            if self.tracking_alg == 'sort':
                cords = np.array(self.detect_image(rgb_image))
            else:
                cords = self.detect_image(rgb_image)
            tracker.update(cords, frame, prediction2=prediction2,
                           time_stamp=time_stamp, demo=True)

            video.write(frame2)

            fps1 = fpsWithTick.get()
            fps_count += fps1
            frame_count += 1
            if frame_count == 0:
                frame_count += 1

            if count % 50 == 0:
                f = open(os.path.join(self.save_root_dir, 'prediction.csv'), "a")
                f2 = open(os.path.join(self.save_root_dir, 'prediction2.csv'), "a")
                writer = csv.writer(f, lineterminator='\n')
                writer2 = csv.writer(f2, lineterminator='\n')
                time_stamp = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
                avg_fps = (fps_count / frame_count)
                fps_count = 0
                frame_count = 0
                prediction.append((time_stamp, tracker.cnt_down, avg_fps))
                writer.writerows(prediction)
                writer2.writerows(prediction2)
                save_movie_dir = os.path.join(self.save_movie_dir, (time_stamp+'.avi'))
                prediction = []
                prediction2 = []

                # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(save_movie_dir, fourcc,
                                        20, (int(cap.get(3)), int(cap.get(4))))
                f.close()

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        cap.release()
        video.release()
        cv2.destroyAllWindows()

    def detect_image(self, frame):
        if self.model == 'yolo':
            img, _, _, _ = letterbox(frame, height=self.img_size)
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            pred = self.net(img)
            # remove boxes < threshold
            pred = pred[pred[:, :, 4] > self.conf_thres]
            cords = []

            if len(pred) > 0:
                # Run NMS on predictions
                detections = non_max_suppression(
                    pred.unsqueeze(0), self.conf_thres, self.nms_thres)[0]
                if detections is not None:

                    # Rescale boxes from 416 to true image size
                    detections[:, :4] = scale_coords(
                        self.img_size, detections[:, :4], frame.shape)

                    # Print results to screen

                    for x1, y1, x2, y2, conf, cls_conf, cls in detections:
                        pt = np.array(
                            [x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])
                        cords.append(pt)

            return cords

        elif self.model == 'ssd':
            x = cv2.resize(frame, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)

            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()

            x = torch.from_numpy(x).permute(2, 0, 1)
            xx = x.unsqueeze(0).to(self.device)
            # if torch.cuda.is_available():
            #    xx = xx.cuda()
            y = self.net(xx)

            detections = y.data
            scale = torch.Tensor([frame.shape[1::-1], frame.shape[1::-1]]).view(4)
            cords = []
            # texts = []
            for i in range(1, 2):
                j = 0
                while detections[0, i, j, 0] >= self.ssd_th:
                    # score = detections[0, i, j, 0]
                    # label_name = labels[i-1]
                    # display_txt = '%s: %.2f' % (label_name, score)
                    # label_name = labels[i-1]
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    cords.append(pt)
                    # texts.append(display_txt)
                    j += 1

            return cords

    def realtime_detection(self, path_to_movie):
        cap = cv2.VideoCapture(path_to_movie)
        basename = os.path.basename(path_to_movie).replace('.mp4', '')
        movie_id = basename[0:4]

        save_movie_path = os.path.join(self.movie_dir, basename+'.mp4')
        print(save_movie_path)
        if self.video:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            video = cv2.VideoWriter(save_movie_path, fourcc,
                                    self.frame_rate, (int(cap.get(3)), int(cap.get(4))))
        height = cap.get(4)
        line_down = int(9*(height/10))
        t1 = threading.Thread(target=self.recall_q2, args=(line_down, height, movie_id, basename))
        #t2 = threading.Thread(target=self.detection_q, args=(line_down, height, movie_id, basename,))
        t1.start()
        # t2.start()
        i = 0

        while(cap.isOpened()):
            i += 1
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.q.append(rgb_image)
            else:
                self.flag_of_realtime = False
                break
            if self.video:
                video.write(frame)
        print(i)

    def recall_q(self):
        while self.flag_of_realtime:
            if self.q:
                newFrame = self.q.popleft()
                if newFrame is not None:
                    self.recallq.append(newFrame)  # 新しいframeはrecallにいれる
                    self.p += 1
                    if len(self.recallq) > 5:
                        self.recallq.popleft()
                    if self.p % 2 == 0 and self.flag_of_recall_q:  # 確率Pで検出を行う
                        cords = self.detect_image(newFrame)
                        if cords is not None:  # 検出ができた時は、recallqの全てframeを処理する
                            self.flag_of_recall_q = False
                            self.flag_of_detection_q = True
                else:
                    continue
#

    def recall_q2(self, line_down, height, movie_id, basename):
        t = time.time()
        LC = self.l/self.frame_rate  # L/C
        Ps = 0.1
        Pd = 1  # Pdは１から減る
        Tw = 10
        self.tracking_alg = 'iou'
        if self.tracking_alg == 'sort':
            tracker = Sort(1, self.max_age, line_down, movie_id,
                           self.image_dir, '', basename)
        else:
            tracker = Iou_Tracker(

                line_down, self.image_dir, movie_id, self.max_age, '', basename)
        # Qはrecallqで、一時的に保存する
        # 検出を行わない時にQに入れる
        i = 0
        flag = False
        while self.flag_of_realtime or self.q:
            if self.q:
                i += 1
                newFrame = self.q.popleft()
                if newFrame is not None:
                    Ran = random.random()
                    if len(self.recallq) < 10:
                        self.recallq.append(newFrame)
                        continue
                    if Ran < Pd:  # 確率Ranで検出
                        # fps1 = self.fpsWithTick.get()
                        # self.fps_count += fps1
                        # self.frame_count += 1
                        cords = self.detect_image(newFrame)  # 変数cordsをそのまま使用 cords＝座標
                        # 動画のどこかに検出された座標の値を表示させる
                        # 検出できたらどこかに残して、動画に出力する
                        if cords:  # 検出ができた時に Pdを1にしてQをdetect
                            Pd = 1
                            self.w = 0
                            # Whileでrecall_qの中のデーターを検出することを繰り返す
                            # detectするのはここだけ
                            while self.recallq:
                                img = self.recallq.popleft()
                                detectQ = self.detect_image(img)  # 座標の位置がリターンされる
                                tracker.update(detectQ, img)

                        else:
                            self.w += 1
                            if self.w >= Tw:
                                Pd = max(Pd - Ps, LC)
                    else:
                        if Tw > len(self.recallq):
                            self.recallq.append(newFrame)  # EnQu(バッファに追加)
                        else:
                            self.recallq.append(newFrame)
                            self.recallq.popleft()
                else:
                    continue
        t = time.time() - t
        print('end_time:{0:0.3f} seconds'.format(t))

    def detection_q(self, line_down, height, movie_id, basename):
        if self.tracking_alg == 'sort':
            tracker = Sort(1, self.max_age, line_down, movie_id,
                           self.image_dir, '', basename)
        else:
            tracker = Iou_Tracker(
                line_down, self.image_dir, movie_id, self.max_age, '', basename)

        while self.flag_of_realtime:
            if self.flag_of_detection_q:
                try:
                    if self.tracking_alg == 'sort':
                        frame = self.recallq.popleft()
                        cords = np.array(self.detect_image(frame))
                        print(cords)
                    else:
                        frame = self.recallq.popleft()
                        cords = self.detect_image(frame)
                        print(cords)
                    tracker.update(cords, frame, fps_eval=self.fps_eval)
                except IndexError:
                    self.flag_of_detection_q = False
                    continue
                self.i += 1
                if self.i > 10:
                    self.flag_of_detection_q = False
                    self.flag_of_recall_q = True
                    self.i = 0
