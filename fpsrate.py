# -*- coding: utf-8 -*-

import cv2


class FpsWithTick(object):
    def __init__(self):
        self._count = 0
        self._oldCount = 0
        self._freq = 1000 / cv2.getTickFrequency()
        self._startTime = cv2.getTickCount()

    def get(self):
        nowTime = cv2.getTickCount()
        diffTime = (nowTime - self._startTime) * self._freq
        self._startTime = nowTime
        fps = (self._count - self._oldCount) / (diffTime / 1000.0)
        self._oldCount = self._count
        self._count += 1
        fpsRounded = round(fps, 1)
        return fpsRounded
