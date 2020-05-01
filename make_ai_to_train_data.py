#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FILE:
  make_slided_train_data.py

CREATER:
  naisy (https://github.com/naisy/

SUMMARY:
  Compared to real time, computer processing is always delayed. Especially, deeplearning takes time for camera processing and prediction processing. Even when the camera works at 120 fps, there is 8.3 ms delay. For 21 fps there is 47.6 ms delay. If prediction is 20 fps, the delay will be added by 50 ms. If frame resize is included, the delay will exceed 100 ms.

  The distance that a 10 km/h (= 2.77 m/s) car body moves to 100 ms is 27.7 cm. The total length of 1/10 RC is about 40 cm, and the total width is about 19 cm. Even if perfect drift data is prepared on a narrow road, it is impossible to drift without crashing if a delay of 100 ms occurs.

  Therefore, prepare the data by intentionally shifting the frame. This is the source code.

  リアルタイムと比べると、コンピュータ処理はいつも遅延します。特にディープラーニングではカメラ処理と予測処理に時間がかかります。120fpsで動作するカメラでも8.3msの遅延があります。21fpsのカメラでは47.6msの遅延になります。予測が20fpsで動作する時はさらに50msの遅延が追加されることになります。フレームのリサイズ処理を含めると、総遅延は100msを超えることになります。

  10km/h(=2.77m/s)の車体が100msに移動する距離は、27.7cmです。1/10 RCの全長は約40cm、全幅は約19cmです。道幅の狭い道路で完璧なドリフトデータを用意しても、100msの遅延が発生するとクラッシュせずにドリフト走行することは不可能です。

  そこで、意図的にフレームをずらしてデータを用意します。これはそのソースコードになります。

LICENSE:
MIT License

Copyright (c) 2019 naisy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import time
import os
import sys
from stat import *
import json
from collections import defaultdict

from shutil import copyfile

def walktree(dir_path, callback):
    """
    Reference:
    https://stackoverflow.com/questions/3204782/how-to-check-if-a-file-is-a-directory-or-regular-file-in-python

    recursively descend the directory tree rooted at top,
    calling the callback function for each regular file
    """
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        mode = os.stat(file_path)[ST_MODE]
        if S_ISDIR(mode):
            # It's a directory, recurse into it
            walktree(file_path, callback)
        elif S_ISREG(mode):
            # It's a file
            if file_name == "meta.json":
                # skip meta file
                #print('{} - skip'.format(file_path))
                pass
            elif file_name.endswith(".json"):
                # call the callback function
                callback(dir_path, file_name)
            else:
                # Unknown file type, print a message
                #print('{} - skip'.format(file_path))
                pass
        else:
            # Unknown file type, print a message
            print('{} - unknown'.format(file_path))
    return

def tupleUpdate(variable, itr, value):
    l = list(variable)
    l[itr] = value
    t = tuple(l)
    return t

def tupleImageUpdate(variable, value):
    l = list(variable)
    l[0]["cam/image_array"] = value
    t = tuple(l)
    return t

class DataListBuilder:
    def __init__(self, src_dir_path):
        self.data_list = defaultdict(list)

        if not os.path.exists(src_dir_path):
            raise ValueError('File not found: '+src_dir_path)

        walktree(src_dir_path, self.addFiles)

        return

    def __del__(self):
        return

    def addFiles(self, dir_path, file_name):
        self.data_list[dir_path].append(file_name)
        return

    def getDataList(self):
        return self.data_list

class DonkeyElementChanger():
    def __init__(self, data_list, dst_dir_path='data_ai'):
        """
        data_list: dataset list. defaultdict (for dirs) with list type (for files).
        dst_dir_path: output dataset dir
        """
        self.data_list = data_list
        self.dst_dir_path = dst_dir_path
        self.src_dir_path = None
        return

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def change(self, st_rate=1.0, th_rate=1.0):
        dir_number     = 0
        file_number    = 0
        is_meta_copy   = False

        for dir_path in self.data_list:
            is_meta_copy      = False
            src_dir_path      = dir_path
            dst_dir_path      = os.path.join(self.dst_dir_path, dir_path[5:])
            dir_number       += 1
            # get num of files in the directory
            data_list_len = len(self.data_list[dir_path])
            file_number = data_list_len
            while(file_number != -1):
                if file_number == 0:
                    file_number -= 1
                    break
                file_name = "record_{}.json".format(file_number)
                file_path = os.path.join(dir_path, file_name)
                file_number -= 1
                if not os.path.exists(file_path):
                    print("Not found: {} - skip".format(file_path))
                    # dataset is empty. read next file
                    continue
                # dataset is exist
                record = self.readJson(file_path)
                if 'user/angle' in record and record["user/angle"] != 0.0:
                    record["user/angle"] = record["pilot/angle"] * (1.0 - st_rate) + record["user/angle"] * st_rate
                else:
                    record["user/angle"] = record["pilot/angle"]
                if 'user/throttle' in record and record["user/throttle"] != 0.0:
                    record["user/throttle"] = record["pilot/throttle"] * (1.0 - st_rate) + record["user/throttle"] * st_rate
                else:
                    record["user/throttle"] = record["pilot/throttle"]

                if record["user/angle"] > 1.0:
                    record["user/angle"] = 1.0
                elif record["user/angle"] < -1.0:
                    record["user/angle"] = -1.0
                if record["user/throttle"] > 1.0:
                    record["user/throttle"] = 1.0
                elif record["user/throttle"] < -1.0:
                    record["user/throttle"] = -1.0

                del record["pilot/angle"]
                del record["pilot/throttle"]

                self.mkdir(dst_dir_path)
                self.writeJson(record, os.path.join(dst_dir_path, file_name))
                # meta copy
                if not is_meta_copy:
                    src = os.path.join(src_dir_path, "meta.json")
                    dst = os.path.join(dst_dir_path, "meta.json")
                    copyfile(src, dst)
                    is_meta_copuy = True
                # image copy
                src = os.path.join(src_dir_path, record["cam/image_array"])
                dst = os.path.join(dst_dir_path, record["cam/image_array"])
                copyfile(src, dst)

                print("{} {} file write".format(dir_number, file_number))

    def readJson(self, file_path):
        json_data = None
        #print(file_path)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def writeJson(self, json_data, file_path):
        #json_string = json.dumps(json_data)
        # If the file name exists, write a JSON string into the file.
        if file_path:
            # Writing JSON data
            with open(file_path, 'w') as f:
                json.dump(json_data, f)

def main():
    USER_STEERING_ASSIST_RATE = 1.0
    USER_THROTTLE_ASSIST_RATE = 1.0

    data_list_builder= DataListBuilder(src_dir_path='data')
    data_list = data_list_builder.getDataList()
    donkey_element_changer = DonkeyElementChanger(data_list=data_list, dst_dir_path='data_ai')
    donkey_element_changer.change(st_rate=USER_STEERING_ASSIST_RATE, th_rate=USER_THROTTLE_ASSIST_RATE)
    return

if __name__ == '__main__':
    main()



