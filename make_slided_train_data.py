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
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
if PY2:
    import Queue
elif PY3:
    import queue as Queue

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

class StreamingArrayList():
    def __init__(self):
        return

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

class DonkeyFrameSlider():
    def __init__(self, data_list, slide_ms=100, dst_dir_path='data_slided'):
        """
        data_list: dataset list. defaultdict (for dirs) with list type (for files).
        slide_ms: frame slide milliseconds. 100 means train record use 100ms older frame image.
        dst_dir_path: output dataset dir
        """
        self.data_list = data_list
        self.slide_ms = slide_ms
        self.dst_dir_path = dst_dir_path
        self.src_dir_path = None
        return

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return

    def slide(self):
        packet         = None # One array. record data, dir_path, file_name, ms, diff_ms, is_updated, itr_flag
        pickup_target  = None # slide target data. from first stream packet
        stream         = [] # record array
        stream_fifo    = [] # temp array. from dataset
        stream_filo    = [] # temp array. from stream
        PKT_RECORD     = 0
        PKT_DIR        = 1
        PKT_FILE       = 2
        PKT_MS         = 3
        PKT_DIFF       = 4
        PKT_UPDATED    = 5
        PKT_ITR        = 6
        ITR_NONE       = 0
        ITR_FAR        = 1
        ITR_NEAR       = 2
        ITR_STOP       = 3
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
                if pickup_target is None:
                    if len(stream) == 0:
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
                        ms = record["milliseconds"]
                        diff_ms = None
                        is_updated = False
                        itr_flag = ITR_NONE
                        pickup_target = (record, dir_path[5:], file_name, ms, diff_ms, is_updated, itr_flag)
                    else:
                        pickup_target = stream.pop(0)
                # here, pickup_target is exist

                # ITR_FAR
                pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_FAR)
                while(pickup_target[6] == ITR_FAR):
                    if file_number == 0:
                        if len(stream) == 0:
                            file_number -= 1
                            pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_NEAR)
                            break
                        else:
                            pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_NEAR)
                            break
                    else:
                        file_name = "record_{}.json".format(file_number)
                        file_path = os.path.join(dir_path, file_name)
                        file_number -= 1
                        if not os.path.exists(file_path):
                            print("Not found: {} - skip".format(file_path))
                            # dataset is empty. read next file
                            pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_FAR)
                            break
                        # dataset is exist
                        record = self.readJson(file_path)
                        ms = record["milliseconds"]
                        diff_ms = None
                        is_updated = False
                        itr_flag = ITR_NONE
                        packet = (record, dir_path[5:], file_name, ms, diff_ms, is_updated, itr_flag)

                    print("{} {} ITR_FAR".format(dir_number, file_number))
                    diff = pickup_target[3] - packet[3]
                    diff_slide_ms = abs(self.slide_ms - (pickup_target[3] - packet[3]))
                    if diff <= 0:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_NEAR)
                        stream_fifo += [packet]
                        print("BROKEN DATA")
                        break
                    if diff < self.slide_ms/2:
                        stream_fifo += [packet]
                        continue
                    if diff > self.slide_ms*1.5:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_NEAR)
                        stream_fifo += [packet]
                        break
                    # diff is ok
                    if not pickup_target[5]: # pickup_target is_updated == False
                        pickup_target = tupleImageUpdate(pickup_target, packet[0]["cam/image_array"])
                        pickup_target = tupleUpdate(pickup_target, PKT_DIFF, pickup_target[3] - packet[3])
                        pickup_target = tupleUpdate(pickup_target, PKT_UPDATED, True)
                        stream_fifo += [packet]
                        continue
                    # pickup_target is_updated == True
                    if abs(self.slide_ms - abs(pickup_target[4])) < diff_slide_ms:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_NEAR)
                        stream_fifo += [packet]
                        break
                    # pickup_target needs update
                    pickup_target = tupleImageUpdate(pickup_target, packet[0]["cam/image_array"])
                    pickup_target = tupleUpdate(pickup_target, PKT_DIFF, pickup_target[3] - packet[3])
                    pickup_target = tupleUpdate(pickup_target, PKT_UPDATED, True)
                    stream_fifo += [packet]

                # ITR_NEAR
                while(pickup_target[6] == ITR_NEAR):
                    print("{} {} ITR_NEAR".format(dir_number, file_number))
                    if len(stream) == 0:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_STOP)
                        break
                    # stream exists
                    packet = stream.pop()
                    diff = pickup_target[3] - packet[3]
                    diff_slide_ms = abs(self.slide_ms - (pickup_target[3] - packet[3]))
                    if diff <= 0:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_STOP)
                        stream_fifo += [packet]
                        print("BROKEN DATA")
                        break
                    if diff < self.slide_ms/2:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_STOP)
                        stream_filo += [packet]
                        break
                    if diff > self.slide_ms*1.5:
                        stream_filo += [packet]
                        continue
                    # diff is ok
                    if not pickup_target[5]: # pickup_target is_updated == False
                        pickup_target = tupleImageUpdate(pickup_target, packet[0]["cam/image_array"])
                        pickup_target = tupleUpdate(pickup_target, PKT_DIFF, pickup_target[3] - packet[3])
                        pickup_target = tupleUpdate(pickup_target, PKT_UPDATED, True)
                        stream_filo += [packet]
                        continue
                    # pickup_target is_updated == True
                    if abs(self.slide_ms - abs(pickup_target[4])) < diff_slide_ms:
                        pickup_target = tupleUpdate(pickup_target, PKT_ITR, ITR_STOP)
                        stream_filo += [packet]
                        break
                    # pickup_target needs update
                    pickup_target = tupleImageUpdate(pickup_target, packet[0]["cam/image_array"])
                    pickup_target = tupleUpdate(pickup_target, PKT_DIFF, pickup_target[3] - packet[3])
                    pickup_target = tupleUpdate(pickup_target, PKT_UPDATED, True)
                    stream_filo += [packet]

                # end
                if not pickup_target[5]:
                    print("{} {} file not write {} {}".format(dir_number, file_number, pickup_target[2], pickup_target[3]))
                    pickup_target = None
                else:
                    self.mkdir(dst_dir_path)
                    self.writeJson(pickup_target[0], os.path.join(dst_dir_path, pickup_target[2]))
                    # meta copy
                    if not is_meta_copy:
                        src = os.path.join(src_dir_path, "meta.json")
                        dst = os.path.join(dst_dir_path, "meta.json")
                        copyfile(src, dst)
                        is_meta_copuy = True
                    # image copy
                    src = os.path.join(src_dir_path, pickup_target[0]["cam/image_array"])
                    dst = os.path.join(dst_dir_path, pickup_target[0]["cam/image_array"])
                    copyfile(src, dst)

                    print("{} {} file write {} {} {}".format(dir_number, file_number, pickup_target[2], pickup_target[3], pickup_target[4]))
                    pickup_target = None

                print("{} {} FILO:{} FIFO:{}".format(dir_number, file_number, len(stream_filo), len(stream_fifo)))
                while len(stream_filo) > 0:
                    stream += [stream_filo.pop()]
                while len(stream_fifo) > 0:
                    stream += [stream_fifo.pop(0)]

    def showData(self):
        dir_count = 0
        for dir_path in self.data_list:
            file_count = 0
            for file_name in self.data_list[dir_path]:
                file_path = os.path.join(dir_path, file_name)
                json_data = self.readJson(file_path)
                print("{} {} {} {} {}".format(dir_count, file_count, dir_path, file_name, json_data["milliseconds"]))
                file_count += 1
            dir_count += 1

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
    data_list_builder= DataListBuilder(src_dir_path='data')
    data_list = data_list_builder.getDataList()
    donkey_frame_slider = DonkeyFrameSlider(data_list=data_list, slide_ms=100, dst_dir_path='data_slided')
    donkey_frame_slider.slide()
    return

if __name__ == '__main__':
    main()


