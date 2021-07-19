 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE:
  convert_v2_to_v1.py

AUTHOR:
  naisy (https://github.com/naisy)

SUMMARY:
  Convert data format from donkeycar 4.3 tub_v2 to donkeycar 3.1 tub_v1.

LICENSE:
  MIT License

Copyright (c) 2021 naisy
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

import os
import json
from collections import OrderedDict
import shutil

def delete_images(dir, delete_id_list):
    for i in delete_id_list:
        file = f'{i}_cam_image_array_.jpg'
        file_path = os.path.join(dir, file)
        delete_image(file_path)


def delete_image(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f'delete {file_path} - ok')
        except Exception as e:
            print(f'delete {file_path} - error!')
    else:
        print(f'delete {file_path} - ng')


def mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    

def copy_to_v1(v2_data_dir, v1_data_dir, catalog_list):
    mkdir(v1_data_dir)
    record_i = 1
    for catalog in catalog_list:
        file_path = os.path.join(v2_data_dir, catalog)
        f = open(file_path, 'r')
        v2_records = f.readlines()
        for v2_record in v2_records:
            try:
                v1_record = OrderedDict()
                json_data = json.loads(v2_record)
                v1_record['cam/image_array']  = f'{record_i}_cam_image_array_.jpg'
                v1_record['user/angle']       = json_data['user/angle']
                v1_record['user/throttle']    = json_data['user/throttle']
                if 'pilot/angle' in json_data.keys():
                    v1_record['pilot/angle']       = json_data['pilot/angle']
                if 'pilot/throttle' in json_data.keys():
                    v1_record['pilot/throttle']       = json_data['pilot/throttle']
                v1_record['user/mode']        = json_data['user/mode']
                v1_record['milliseconds']     = json_data['_timestamp_ms']

                file = f'record_{record_i}.json'
                file_path = os.path.join(v1_data_dir, file)
                writeJson(v1_record, file_path)
                # copy image
                src = os.path.join(v2_data_dir, 'images', json_data['cam/image_array'])
                dst = os.path.join(v1_data_dir, v1_record['cam/image_array'])
                shutil.copyfile(src, dst)
                print(f'{file_path} - saved')
                record_i += 1
            except FileNotFoundError as e1:
                print(f'skip deleted record: {catalog} - {json_data}')
            except Exception as e2:
                print(f'skip broken record: {catalog} - {json_data}')
                print(e2)


def read_manifest(dir):
    '''
    manifest.json is not json file.
    so we need to pick up correct lines.
    '''
    file = 'manifest.json'
    file_path = os.path.join(dir, file)
    f = open(file_path, 'r')
    datalist = f.readlines()
    return datalist

def writeJson(json_data, file_path):
    #json_string = json.dumps(json_data)
    # If the file name exists, write a JSON string into the file.
    if file_path:
        # Writing JSON data
        with open(file_path, 'w') as f:
            json.dump(json_data, f)

def main():
    '''
    1. delete image
    2. copy exist image and value
    '''
    v2_data_dir = './data'
    manifest_lines = read_manifest(v2_data_dir)
    json_data = json.loads(manifest_lines[4])
    catalog_list = sorted(json_data['paths'])
    delete_id_list = sorted(json_data['deleted_indexes'])
    image_dir = os.path.join(v2_data_dir, 'images')
    delete_images(image_dir, delete_id_list)
    
    v1_data_dir = './data_v1'
    copy_to_v1(v2_data_dir, v1_data_dir, catalog_list)


if __name__ == '__main__':
    main()
