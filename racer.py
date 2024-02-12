"""
Overview:
    TensorRT/Tensorflow/Tensorflow Lite Racer for the DonkeyCar Virtual Race.
Usage:
    racer.py (--host=<ip_address>) (--name=<car_name>) (--model=<model_path>) (--delay=<seconds>)
Example:
    python racer.py --host=127.0.0.1 --name=naisy --model=linear.engine --delay=0.2

Supported models:
    TensorRT 8:       linear.engine
    Tensorflow:       linear.h5
    Tensorflow Lite:  linear.tflite
"""

import socket
import cv2
import select
import time
from docopt import docopt
import json
import logging
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import queue as Queue
import threading
import re
import traceback
from pynput import keyboard
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)  # NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50


# モデルロードと推論を行うクラス（RaceClientから推論部分を抜粋・変更）
class InferenceModel():
    def __init__(self, model_path):
        if model_path.endswith('.engine'):
            self.model = TRTModel(model_path=model_path)
        elif model_path.endswith('.h5'):
            self.model = TFModel(model_path=model_path)
        elif model_path.endswith('.tflite'):
            self.model = TFLiteModel(model_path=model_path)

    def infer(self, image):
        x = self.model.preprocess(image)
        [throttle, steering] = self.model.infer(x)
        return throttle[0], steering[0]

# 画像ファイルから推論し、結果を動画に保存する関数
def process_images_to_video(image_dir, model_path, output_video, start_frame=0, end_frame=1000, fps=20):
    model = InferenceModel(model_path=model_path)

    # MP4形式で動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 コーデック
    out = cv2.VideoWriter(output_video, fourcc, fps, (320, 24))  # 画像サイズに応じて変更
    count = 0
    start_time = time.time()
    
    for i in range(start_frame, end_frame + 1):
        image_path = os.path.join(image_dir, f"{i}_cam_image_array_.jpg")
        if not os.path.exists(image_path):
            print(f"Image {image_path} not found.")
            continue

        image = cv2.imread(image_path)
        throttle, steering = model.infer(image)
        if throttle > 1.0:
            throttle = 1.0
        elif throttle < -1.0:
            throttle = 1.0
        if steering > 1.0:
            steering = 1.0
        elif steering < -1.0:
            steering = -1.0

        #print(f"{steering}, {throttle}")
        
        # 推論結果を画像に描画（オプション）
        image = draw_circle(image, throttle, steering)

        # 動画にフレームを追加
        out.write(image)
        if count == 0:
            start_time = time.time()
        count += 1

    out.release()
    print(f"Video saved to {output_video}")
    end_time = time.time()
    print(f"fps: {(count/(end_time - start_time)):0.2f}")

def draw_circle(image, throttle, steering):
    # 画像の高さと幅を取得
    height, width = image.shape[:2]

    # throttleとsteeringの値に基づいて、丸の位置を計算
    if steering <= -1.0 and throttle <= -1.0:
        center_coordinates = (int(width * 0.1), int(height * 0.9))  # 左下
    elif steering == 0.0 and throttle == 0.0:
        center_coordinates = (int(width * 0.5), int(height * 0.5))  # 真ん中
    elif steering >= 1.0 and throttle >= 1.0:
        center_coordinates = (int(width * 0.9), int(height * 0.1))  # 右上
    else:
        # 標準化されていない値の場合、中心を基準に位置を計算
        center_coordinates = (int(width * (0.5 + steering * 0.5)), int(height * (0.5 - throttle * 0.5)))

    # throttleの値に応じて、丸の色を選択
    if throttle >= 0:
        color = (0, 255, 0)  # 緑色
    else:
        color = (0, 0, 255)  # 赤色

    # 円を描画
    radius = 3  # 丸の半径
    thickness = -1  # 塗りつぶし
    image = cv2.circle(image, center_coordinates, radius, color, thickness)

    return image


class TRTModel():

    def __init__(self, model_path):

        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None

        self.runtime = trt.Runtime(TRT_LOGGER)
        MODEL_TYPE = 'linear'

        self.engine = self.load_engine(model_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def close(self):
        self.context = None

    def load_engine(self, model_path):
        # load tensorrt model from file
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        print(f'Load model from {model_path}.')
        return engine

    def save_engine(self, engine, model_path):
        # save tensorrt model to file
        serialized_engine = engine.serialize()
        with open(model_path, "wb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            f.write(engine.serialize())
        print(f'Save model to {model_path}.')

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_memory = cuda.pagelocked_empty(size, dtype)
            device_memory = cuda.mem_alloc(host_memory.nbytes)
            bindings.append(int(device_memory))
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_memory, device_memory))
            else:
                outputs.append(HostDeviceMemory(host_memory, device_memory))

        return inputs, outputs, bindings, stream

    def preprocess(self, image):
        # image: rgb image

        # RGB convert from [0, 255] to [0.0, 1.0]
        x = image.astype(np.float32) / 255.0
        # HWC to CHW format
        x = x.transpose((2, 0, 1)) # keras -> ONNX -> TRT8, don't need HWC to CHW. model inputs uses HWC.
        # Flatten it to a 1D array.
        x = x.reshape(-1)
        #x = x.ravel()
        return x

    def infer(self, x, batch_size=1):
        # The first input is the image. Copy to host memory.
        image_input = self.inputs[0]
        np.copyto(image_input.host_memory, x)

        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device_memory, inp.host_memory, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host_memory, out.device_memory, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        #print(self.outputs)
        return [out.host_memory for out in self.outputs]

class TFLiteModel():

    def __init__(self, model_path):
        self.interpreter = self.load_model(model_path)

    def close(self):
        return

    def load_model(self, model_path):
        print(f'Load model from {model_path}.')
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        return interpreter

    def preprocess(self, image):
        # image: rgb image

        # RGB convert from [0, 255] to [0.0, 1.0]
        x = image.astype(np.float32) / 255.0
        # Flatten it to a 1D array.
        #x = x.reshape(-1)
        #x = x.ravel()
        return x
    
    def infer(self, x, batch_size=1):
        self.interpreter.set_tensor(self.input_details[0]['index'], [x])
        self.interpreter.invoke()
        steering = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        throttle = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        return [throttle, steering]

class TFModel():

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def close(self):
        return

    def load_model(self, model_path):
        print(f'Load model from {model_path}.')
        model = keras.models.load_model(model_path)

        return model

    def preprocess(self, image):
        # image: rgb image

        # RGB convert from [0, 255] to [0.0, 1.0]
        x = image.astype(np.float32) / 255.0
        # Flatten it to a 1D array.
        #x = x.reshape(-1)
        x = x[None, :, :, :]
        #x = x.ravel()
        return x
    
    def infer(self, x, batch_size=1):
        outputs = self.model(x, training=False)
        steering = outputs[0][0].numpy() # EagerTensor to numpy
        throttle = outputs[1][0].numpy()
        return [throttle, steering]


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--model'].endswith('.engine'):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        from collections import namedtuple
        global HostDeviceMemory
        global TRT8
        global TRT7
        global TRT_LOGGER
        HostDeviceMemory = namedtuple('HostDeviceMemory', 'host_memory device_memory')
        TRT8 = 8
        TRT7 = 7
        TRT_LOGGER = trt.Logger()

    elif args['--model'].endswith('.h5'):
        import tensorflow as tf
        import tensorflow.keras as keras
    elif args['--model'].endswith('.tflite'):
        import tensorflow as tf

    main(host=args['--host'], name=args['--name'], model_path=args['--model'], delay=args['--delay'])

    #image_dir = 'data/images'
    #model_path = args['--model']
    #output_video = 'output.mp4'
    #process_images_to_video(image_dir, model_path, output_video, start_frame=0, end_frame=6000, fps=60)
