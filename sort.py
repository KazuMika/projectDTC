# -*- coding: utf-8 -*-
from __future__ import print_function
from numba import jit
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment as linear_assignment
import time
from filterpy.kalman import KalmanFilter
import cv2
from fpsrate import FpsWithTick
from utils.count_utils import convert_to_latlng
import os
import warnings

warnings.simplefilter('ignore')


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
                             0, 0, 0, 1, 0, 0, 0],  [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
                             0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.done = False
        self.x, self.y = 0, 0
        self.pre_x, self.pre_y = 0, 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def center_cord(self):
        bbox = convert_x_to_bbox(self.kf.x)[0]
        x = int((bbox[2]+bbox[0])/2)
        h = bbox[3]-bbox[1]
        y = int((bbox[1]+(h*7/8)))
        return x, y


class Sort(object):
    def __init__(self, max_age=2, min_hits=3, line_down=None, movie_id='', save_image_dir=None, movie_date='', basename='', save_movie_dir=None):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.line_down = line_down
        self.cnt_down = 0
        self.movie_id = movie_id
        self.save_image_dir = save_image_dir
        self.save_movie_dir = save_movie_dir
        self.frame_count = 0
        self.fps_count = 0
        self.fpsWithTick = FpsWithTick()
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.movie_date = movie_date
        self.basename = basename

    def going_down(self, pre_y, y, frame=None, gpss=None, gps_count=None, gps_list=None, visualize=None, prediction2=None, demo=False, time_stamp=None, fps_eval=None):
        if y > self.line_down and pre_y <= self.line_down:
            self.cnt_down += 1
            if fps_eval:
                return True

            print('test')

            cv2.imwrite(os.path.join(self.save_image_dir, self.basename+self.movie_date +
                                     "_{0:04d}_{1:03d}.jpg".format(self.frame_count, self.cnt_down)), frame)
            print('count:{}'.format(self.cnt_down))
            if demo:
                img_name = time_stamp + \
                    '_{0:04d}.jpg'.format(
                        self.cnt_down)
                str_down = 'COUNT:' + str(self.cnt_down)
                cv2.putText(frame, str_down, (10, 70), self.font,
                            2.0, (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(frame, str_down, (10, 70), self.font,
                            2.0, (255, 255, 255), 8, cv2.LINE_AA)
                cv2.putText(frame, time_stamp, (900, 40), self.font,
                            1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, time_stamp, (900, 40), self.font,
                            1.0, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.imwrite(
                    self.save_image_dir+img_name, frame)
                prediction2.append(
                    (time_stamp, self.cnt_down))
            else:
                cv2.imwrite(os.path.join(self.save_image_dir, self.basename+self.movie_date +
                                         "_{0:04d}_{1:03d}.jpg".format(self.frame_count, self.cnt_down)), frame)
            if visualize:
                try:
                    lat = gpss[gps_count].split(',')[0][1:]
                    lng = gpss[gps_count].split(',')[1]
                    lat, lng = convert_to_latlng(lat, lng)
                    print(lat, lng)
                    date_gps = self.movie_date + \
                        "_{0:04d}".format(self.cnt_down)
                except:
                    lat, lng = 0, 0
                    print('gregaergargag')
                    date_gps = self.movie_date + \
                        "_{0:04d}".format(self.cnt_down)
                gps_list.append([lat, lng, date_gps])
            return True
        else:
            return False

    def update(self,
               dets,
               frame=None,
               gpss=None,
               gps_count=None,
               visualize=False,
               gps_list=None,
               prediction2=None,
               time_stamp=None,
               demo=False,
               fps_eval=False):

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            self.trackers[t].pre_x, self.trackers[t].pre_y = self.trackers[t].center_cord()
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # trk.pre_x, trk.pre_y = trk.center_cord()
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            h = dets[i][3]-dets[i][1]
            y = int((dets[i][1]+(h*7/8)))
            if y < self.line_down:
                trk = KalmanBoxTracker(dets[i, :])
                self.trackers.append(trk)
        i = len(self.trackers)

        if fps_eval:
            for trk in reversed(self.trackers):
                trk.x, trk.y = trk.center_cord()
                trk.done = self.going_down(trk.pre_y, trk.y, fps_eval=fps_eval)
                d = trk.get_state()[0].astype(np.int)
                i -= 1
                if(trk.time_since_update > self.max_age) or trk.done:
                    self.trackers.pop(i)

            self.fps1 = self.fpsWithTick.get()
            self.fps_count += self.fps1
            self.frame_count += 1

            if self.frame_count == 0:
                self.frame_count += 1
        else:
            for trk in reversed(self.trackers):
                trk.x, trk.y = trk.center_cord()
                d = trk.get_state()[0].astype(np.int)
                i -= 1
                cv2.circle(frame, (trk.x, trk.y), 3, (0, 0, 126), -1)
                cv2.rectangle(
                    frame, (d[0], d[1]), (d[2], d[3]), (0, 252, 124), 2)

                cv2.rectangle(frame, (d[0], d[1] - 20),
                              (d[0] + 170, d[1]), (0, 252, 124), thickness=2)
                cv2.rectangle(frame, (d[0], d[1] - 20),
                              (d[0] + 170, d[1]), (0, 252, 124), -1)
                cv2.putText(frame, str(trk.id+1) + " " + str(trk.time_since_update)+" ",
                            (d[0], d[1] - 5), self.font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                str_down = 'COUNT:' + str(self.cnt_down+1)
                cv2.line(frame, (0, self.line_down),
                         (int(frame.shape[1]), self.line_down), (255, 0, 0), 2)
                cv2.putText(frame, str_down, (10, 70), self.font,
                            2.5, (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(frame, str_down, (10, 70), self.font,
                            2.5, (255, 255, 255), 8, cv2.LINE_AA)
                trk.done = self.going_down(
                    trk.pre_y, trk.y, frame, gpss, gps_count,  gps_list, visualize, prediction2, demo=demo, time_stamp=time_stamp, fps_eval=fps_eval)

                if(trk.time_since_update > self.max_age) or trk.done:
                    self.trackers.pop(i)


@jit
def iou(boxA, boxB):
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
    return iou


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h  # scale is just area
    r = w/float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    # print('iou_matrix', iou_matrix.shape, iou_matrix)
    # print('matched_indices', matched_indices.shape, matched_indices)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] <= iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


if __name__ == '__main__':
    sequences = ['PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'ETH-Bahnhof', 'ETH-Sunnyday',
                 'ETH-Pedcross2', 'KITTI-13', 'KITTI-17', 'ADL-Rundle-6', 'ADL-Rundle-8', 'Venice-2']
    phase = 'train'
    total_time = 0.0
    total_frames = 0

    for seq in sequences:
        mot_tracker = Sort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt('data/%s/det.txt' %
                              (seq), delimiter=',')  # load detections
        with open('output/%s.txt' % (seq), 'w') as out_file:
            print("Processing %s." % (seq))
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                          (frame, d[4], d[0], d[1], d[2]-d[0], d[3]-d[1]), file=out_file)
