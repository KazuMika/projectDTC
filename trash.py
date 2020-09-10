# -*- coding: utf-8 -*-


class Trash:
    def __init__(self, t_id, cords, center, max_age):
        self.id = t_id
        self.cords = cords
        self.done = False
        self.state = True
        self.age = 0
        self.max_age = max_age
        self.center = center
        self.pre_center = None

    def updateCoords(self, cords, center):
        self.age = 0
        self.cords = cords
        self.pre_center = self.center
        self.center = center

    def going_DOWN(self, line_down):
        if self.pre_center is not None:
            if self.center[1] > line_down and self.pre_center[1] <= line_down:
                return True
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
