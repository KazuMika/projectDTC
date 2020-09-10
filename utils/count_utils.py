# -*- coding: utf-8 -*-
import os


def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)


def convert_to_latlng(lat, lng):
    lat = lat.split('.')
    lng = lng.split('.')
    lat[2] = lat[2][:2] + '.' + lat[2][2]
    lng[2] = lng[2][:2] + '.' + lng[2][2]
    print(lat[2])
    print(lng[2])

    lat = (float(lat[2])/3600) + (int(lat[1]) / 60) + int(lat[0])
    lng = (float(lng[2])/3600) + (int(lng[1]) / 60) + int(lng[0])
    return lat, lng
