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

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)  # NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50


class RaceClient:

    def __init__(self, host, port, model_path, delay, car_conf, scene_conf, cam_conf, socket_read_hz=20, name='naisy'):
        self.msg = None
        self.poll_socket_sleep_sec = 1.0/socket_read_hz
        self.car_conf = car_conf
        self.scene_conf = scene_conf
        self.cam_conf = cam_conf
        self.delay = float(delay)
        self.accumulated_delay = 0.0
        self.last_interval = 0.0
        self.simulator_timing_ok = False
        self.name = name
        self.lock = threading.Lock()
        self.count = 0
        self.delay_waiting = False
        self.lap_start_time = None
        self.lap_first_start_time = 0
        self.lap_end_time = 0
        self.last_lap_time = None
        self.best_lap_time = None
        self.lap_time_queue = Queue.Queue(maxsize=400)
        self.lap_counter = 0

        

        # the aborted flag will be set when we have detected a problem with the socket
        # that we can't recover from.
        self.aborted = False

        ### debug
        self.last_image_received_time = None
        self.current_image_received_time = None
        self.ok_frame_counter = 0
        self.ng_frame_counter = 0
        self.last_fps_time = None

        self.queue_size_limit = 10

        self.recv_queue = Queue.Queue(maxsize=self.queue_size_limit)
        self.wait_send_queue = Queue.Queue(maxsize=self.queue_size_limit)
        self.send_queue = Queue.Queue(maxsize=self.queue_size_limit)
        self.image_queue = Queue.Queue(maxsize=self.queue_size_limit)
        
        if model_path.endswith('.engine'):
            self.model = TRTModel(model_path=model_path)
        elif model_path.endswith('.h5'):
            self.model = TFModel(model_path=model_path)
        elif model_path.endswith('.tflite'):
            import tensorflow as tf
            self.model = TFLiteModel(model_path=model_path)

        # connect to unity simulator
        self.connect(host, port)
        # start reading socket
        self.th = threading.Thread(target=self.read_socket, args=(self.sock,))
        self.th.start()

        # keyboard listener
        self.press_any_key = False
        self.press_ctrl = False

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.hotkeys = keyboard.GlobalHotKeys({
                '<ctrl>+c':self.on_activate_ctrl_c})
        self.hotkeys.start()

    def connect(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # connecting to the server 
        print(f'connecting to {host}:{port}')
        self.sock.connect((host, port))
        self.is_socket_read = True


    def send_now(self, msg):
        print("sending now:", msg)
        self.sock.sendall(msg.encode("utf-8"))


    def stop(self):
        self.is_socket_read = False
        self.th.join()
        self.sock.close()
        while not self.image_queue.empty():
            q = self.image_queue.get(block=False)
            self.image_queue.task_done()
        while not self.wait_send_queue.empty():
            q = self.wait_send_queue.get(block=False)
            self.wait_send_queue.task_done()
        while not self.send_queue.empty():
            q = self.send_queue.get(block=False)
            self.send_queue.task_done()
        while not self.recv_queue.empty():
            q = self.recv_queue.get(block=False)
            self.recv_queue.task_done()
        while not self.lap_time_queue.empty():
            q = self.lap_time_queue.get(block=False)
            print(q)
            self.lap_time_queue.task_done()
        try:
            self.hotkeys.join()
        except RuntimeError as e:
            pass
        try:
            self.listener.join()
        except RuntimeError as e:
            pass
        self.model.close()


    def replace_float_notation(self, string):
        """
        Replace unity float notation for languages like
        French or German that use comma instead of dot.
        This convert the json sent by Unity to a valid one.
        Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2
        :param string: (str) The incorrect json string
        :return: (str) Valid JSON string
        """
        regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
        regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'

        for regex in [regex_french_notation, regex_end]:
            matches = re.finditer(regex, string, re.MULTILINE)

            for match in matches:
                num = match.group('num').replace(',', '.')
                string = string.replace(match.group('num'), num)
        return string


    def read_socket(self, sock):
        sock.setblocking(False)
        inputs = [ sock ]
        outputs = [ sock ]
        partial = []

        while self.is_socket_read:
            # without this sleep, I was getting very consistent socket errors
            # on Windows. Perhaps we don't need this sleep on other platforms.
            time.sleep(self.poll_socket_sleep_sec)

            if True: #try:
                # test our socket for readable, writable states.
                readable, writable, exceptional = select.select(inputs, outputs, inputs)

                for s in readable:
                    # print("waiting to recv")
                    try:
                        data = s.recv(1024 * 64)
                    except ConnectionAbortedError:
                        print("socket connection aborted")
                        self.is_socket_read = False
                        break

                    # we don't technically need to convert from bytes to string
                    # for json.loads, but we do need a string in order to do
                    # the split by \n newline char. This seperates each json msg.
                    data = data.decode("utf-8")
                    msgs = data.split("\n")

                    for m in msgs:
                        if len(m) < 2:
                            continue
                        last_char = m[-1]
                        first_char = m[0]
                        # check first and last char for a valid json terminator
                        # if not, then add to our partial packets list and see
                        # if we get the rest of the packet on our next go around.                
                        if first_char == "{" and last_char == '}':
                            # Replace comma with dots for floats
                            # useful when using unity in a language different from English
                            m = self.replace_float_notation(m)
                            j = json.loads(m)
                            self.on_msg_recv(j)
                        else:
                            partial.append(m)
                            if last_char == '}':
                                if partial[0][0] == "{":
                                    assembled_packet = "".join(partial)
                                    assembled_packet = self.replace_float_notation(assembled_packet)
                                    try:
                                        j = json.loads(assembled_packet)
                                    except:
                                        ### reset delay calibration ###
                                        self.simulator_timing_ok = False
                                        self.accumulated_delay = 0.0
                                        ### output error logs ###
                                        traceback.print_exc()
                                        print(partial)
                                        print("######## skip broken packets ########")
                                        partial = []
                                        continue
                                    self.on_msg_recv(j)
                                else:
                                    print("failed packet.")
                                partial.clear()

                for s in writable:
                    now = time.time()
                    while not self.send_queue.empty():
                        q = self.send_queue.get(block=False)
                        if now - q['time'] >= q['delay']:
                            print("sending", q['data'])
                            s.sendall(q['data'].encode("utf-8"))
                            self.send_queue.task_done()
                        else:
                            self.wait_send_queue.put(q)
                    while not self.wait_send_queue.empty():
                        q = self.wait_send_queue.get(block=False)
                        # back to the send_queue
                        self.send_queue.put(q)
                        self.wait_send_queue.task_done()

                if len(exceptional) > 0:
                    print("problems w sockets!")

    def on_msg_recv(self, msg):
        if 'msg_type' in msg:
            logger.debug(f'got {msg["msg_type"]}')
        else:
            logger.debug(f'Unknown: {msg}')
            return

        if msg['msg_type'] == "scene_selection_ready":
            # load scene
            self.scene_config_to_send_queue()

        elif msg['msg_type'] == "car_loaded":
            logger.debug("car_loaded")
            self.car_loaded = True

            self.cam_config_to_send_queue()
            self.car_config_to_send_queue()
        
        elif msg['msg_type'] == "telemetry":
            imgString = msg["image"]
            ### to RGB
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image = np.asarray(image)

            ### interval counter
            interval = 0.05 # initialize
            self.current_image_received_time = time.time()
            if self.last_image_received_time is None:
                q = {'data': image, 'time':time.time(), 'delay': 0.0}
                print(f'receive image: {self.current_image_received_time:10.7f}')
                self.last_fps_time = time.time()
            else:
                interval = self.current_image_received_time - self.last_image_received_time

                if not self.simulator_timing_ok:
                    ### simulator interval calibration ###
                    if interval < 0.0505 and interval >= 0.04995:
                        if self.last_interval < 0.0505 and self.last_interval >= 0.04995:
                            self.simulator_timing_ok = True
                else:
                    ### accumulated delay ###
                    self.accumulated_delay += interval - 0.05
                self.last_interval = interval

                q = {'data': image, 'time':time.time(), 'delay': self.accumulated_delay}
                if interval <= 0.03 or interval >= 0.07:
                    print(f'receive image: {self.current_image_received_time:10.7f}, interval: {interval:.18f} - NG')
                    self.ng_frame_counter += 1
                else:
                    print(f'receive image: {self.current_image_received_time:10.7f}, interval: {interval:.18f}')
                    self.ok_frame_counter += 1
            self.last_image_received_time = self.current_image_received_time

            ### fps counter
            fps_time = self.current_image_received_time - self.last_fps_time
            if fps_time >= 10.0:
                print("----------------------------------------")
                print(f'fps: {(self.ng_frame_counter + self.ok_frame_counter)/fps_time:3.5f}, ok: {self.ok_frame_counter}, ng: {self.ng_frame_counter}')
                print("----------------------------------------")
                self.last_fps_time = time.time()
                self.ng_frame_counter = 0
                self.ok_frame_counter = 0

            ### image show
            #cv2.imshow("frame", image)
            #cv2.waitKey(1)
            ### send control to simulator
            self.lock.acquire()
            if not self.image_queue.empty():
                self.image_queue.get(block=False)
                self.image_queue.task_done()
                print("drop old image")
            self.image_queue.put(q)
            self.lock.release()
        elif msg['msg_type'] == "collision_with_starting_line":
            print(f'collision_with_starting_line: {msg}')
            t = time.time()
            if self.lap_start_time is None:
                self.lap_start_time = t
                self.lap_first_start_time = t
            else:
                self.lap_end_time = t
                self.last_lap_time = t - self.lap_start_time
                self.lap_start_time = t
                is_best = False
                if self.best_lap_time is None:
                    self.best_lap_time = self.last_lap_time
                elif self.last_lap_time < self.best_lap_time:
                    self.best_lap_time = self.last_lap_time
                    is_best = True
                q = {'lap':self.lap_counter, 'lap_time': self.last_lap_time, 'best': is_best}
                self.lap_time_queue.put(q)

                self.lap_counter += 1
            return


    def run_model(self):
        if not self.image_queue.empty():
            start_run_time = time.time()
            print(f'empty count: {self.count}')
            self.count = 0
            self.lock.acquire()
            q = self.image_queue.get(block=False)
            self.lock.release()
            x = q['data']
            x = self.model.preprocess(x)
            start_time = time.time()
            [throttle, steering] = self.model.infer(x)
            end_time = time.time()
            print(f'prediction time: {end_time - start_time:10.7f}')
            self.image_queue.task_done()

            if throttle[0] > 0.95:
                throttle[0] = 1.0
            elif throttle[0] < -1.0:
                throttle[0] = -1.0

            # body color change
            color = self.color_make(throttle[0])
            left_arrow, right_arrow = self.lr_make(steering[0])
            t = time.time()

            if not self.press_any_key:
                name = f'{left_arrow.rjust(3)}{self.name} press any key {throttle[0]:0.2f}{right_arrow.ljust(3)}'
            elif not self.simulator_timing_ok:
                if t - self.lap_first_start_time <= 3.0: # 3.0s
                    name = f'{left_arrow.rjust(3)}{self.name} START {throttle[0]:0.2f}{right_arrow.ljust(3)}'
                elif t - self.lap_end_time <= 3.0: # 3.0s
                    name = f'{self.name} lap:{self.last_lap_time:0.2f})'
                else:
                    name = f'{left_arrow.rjust(3)}{self.name} delay calibrating {throttle[0]:0.2f}{right_arrow.ljust(3)}'
            else:
                if t - self.lap_first_start_time <= 3.0: # 3.0s
                    name = f'{left_arrow.rjust(3)}{self.name} START {throttle[0]:0.2f}{right_arrow.ljust(3)}'
                elif t - self.lap_end_time <= 3.0: # 3.0s
                    name = f'{left_arrow.rjust(3)}{self.name} lap:{self.last_lap_time:0.2f} {throttle[0]:0.2f}{right_arrow.ljust(3)}'
                else:
                    name = f'{left_arrow.rjust(3)}{self.name} {self.accumulated_delay:0.7f} {throttle[0]:0.2f}{right_arrow.ljust(3)}'
            car_conf = {"body_style" : "donkey", 
                        "body_rgb" : color,
                        "car_name" : name,
                        "font_size" : 25}

            self.car_config_to_send_queue(conf=car_conf, delay=self.delay-q['delay'])
            if self.press_any_key:
                self.controls_to_send_queue(steering[0], throttle[0], delay=self.delay-q['delay'])
            print(f"set delay: {self.delay-q['delay']}")
            end_run_time = time.time()
            print(f'run_model time: {end_run_time - start_run_time:10.7f}')
        else:
            self.count += 1

    def color_make(self, value):
        """
        Rainbow color maker.
        value: -1.0 to 1.0
        abs(value) 0.0: blue
        abs(value) 0.5: green
        abs(value) 1.0: red
        """
        value = abs(value)
        if value > 1:
            value = 1

        c = int(255*value)
        c1 = int(255*(value*2-0.5))
        c05 = int(255*value*2)
        if c > 255:
            c = 255
        elif c < 0:
            c = 0
        if c1 > 255:
            c1 = 255
        elif c1 < 0:
            c1 = 0
        if c05 > 255:
            c05 = 255
        elif c05 < 0:
            c05 = 0

        if 0 <= value and value < 0.5:
            color = (0,c05,255-c05) # blue -> green
        elif 0.5 <= value and value <= 1.0:
            color = (c1,c05-c1,0) # green -> red
        elif 1.0 < value:
            color = (255,0,0) # red

        return color


    def lr_make(self, value):
        """
        Rainbow color maker.
        value: -1.0 to 1.0
        abs(value) 0.0: blue
        abs(value) 0.5: green
        abs(value) 1.0: red
        """
        if value > 1:
            value = 1
        elif value < -1:
            value = -1

        left_arrow = '<'
        right_arrow = '>'

        if value < 0:
            if value <= -0.7:
                left_arrow = '<<<'
                right_arrow = ''
            elif value <= -0.3:
                left_arrow = '<<'
                right_arrow = ''
            elif value <= -0.125:
                left_arrow = '<'
                right_arrow = ''
            else:
                left_arrow = '<'
                right_arrow = '>'
        elif value >= 0:
            if value >= 0.7:
                left_arrow = ''
                right_arrow = '>>>'
            elif value >= 0.3:
                left_arrow = ''
                right_arrow = '>>'
            elif value >= 0.125:
                left_arrow = ''
                right_arrow = '>'
            else:
                left_arrow = '<'
                right_arrow = '>'

        return left_arrow, right_arrow


    def scene_config_to_send_queue(self, conf=None):
        logger.debug("scene_config_to_send_queue")
        if conf is None:
            conf = self.scene_conf
        msg = json.dumps(conf)
        q = {'data':msg, 'time':time.time(), 'delay': 0.0}
        self.send_queue.put(q)

    def car_config_to_send_queue(self, conf=None, delay=0.0):
        logger.debug("car_config_to_send_queue")
        if conf is None:
            conf = self.car_conf
        if "body_style" in conf:
            msg = {'msg_type': 'car_config',
                   'body_style': conf["body_style"],
                   'body_r' : str(conf["body_rgb"][0]),
                   'body_g' : str(conf["body_rgb"][1]),
                   'body_b' : str(conf["body_rgb"][2]),
                   'car_name': conf["car_name"],
                   'font_size' : str(conf["font_size"])}
            msg = json.dumps(msg)
            q = {'data':msg, 'time':time.time(), 'delay': delay}
            self.send_queue.put(q)

    def cam_config_to_send_queue(self, conf=None):
        logger.debug("cam_config_to_send_queue")
        if conf is None:
            conf = self.cam_conf
        """ Camera config
            set any field to Zero to get the default camera setting.
            offset_x moves camera left/right
            offset_y moves camera up/down
            offset_z moves camera forward/back
            rot_x will rotate the camera
            rot_y will rotate the camera
            rot_z will rotate the camera
            with fish_eye_x/y == 0.0 then you get no distortion
            img_enc can be one of JPG|PNG|TGA
        """
        msg = {"msg_type" : "cam_config",
               "fov" : str(conf["fov"]),
               "fish_eye_x" : str(conf["fish_eye_x"]),
               "fish_eye_y" : str(conf["fish_eye_y"]),
               "img_w" : str(conf["img_w"]),
               "img_h" : str(conf["img_h"]),
               "img_d" : str(conf["img_d"]),
               "img_enc" : str(conf["img_enc"]),
               "offset_x" : str(conf["offset_x"]),
               "offset_y" : str(conf["offset_y"]),
               "offset_z" : str(conf["offset_z"]),
               "rot_x" : str(conf["rot_x"]),
               "rot_y" : str(conf["rot_y"]),
               "rot_z" : str(conf["rot_z"])}
        msg = json.dumps(msg)
        q = {'data':msg, 'time':time.time(), 'delay': 0.0}
        self.send_queue.put(q)

    def controls_to_send_queue(self, steering, throttle, delay=0.0):
        logger.debug("controls_to_send_queue: {steering}, {throttle}")
        p = {"msg_type" : "control",
             "steering" : steering.__str__(),
             "throttle" : throttle.__str__(),
             "brake" : "0.0"}
        msg = json.dumps(p)
        q = {'data':msg, 'time':time.time(), 'delay': delay}
        self.send_queue.put(q)

    def on_press(self, key):
        try:
            self.press_any_key = True
            print('alphanumeric key {0} pressed'.format(key.char))
            if key.char == 'c' and self.press_ctrl:
                print('<ctrl>+c pressed')
                print('keyboard lister stopped')
                raise KeyboardInterrupt()
        except AttributeError:
            print('special key {0} pressed'.format(key))
            if key == keyboard.Key.ctrl:
                self.press_ctrl = True

    def on_release(self, key):
        print('{0} released'.format(key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False
        if key == keyboard.Key.ctrl:
            self.press_ctrl = False

    def on_activate_ctrl_c(self):
        print('<ctlr>+c pressed')
        print('hotkey listener stopped')
        raise KeyboardInterrupt()


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
        #x = x.transpose((2, 0, 1)) # keras -> ONNX -> TRT8, don't need HWC to CHW. model inputs uses HWC.
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
        # HWC to CHW format
        #x = x.transpose((2, 0, 1)) # keras -> ONNX -> TRT8, don't need HWC to CHW. model inputs uses HWC.
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
        # HWC to CHW format
        #x = x.transpose((2, 0, 1)) # keras -> ONNX -> TRT8, don't need HWC to CHW. model inputs uses HWC.
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


def main(host, name, model_path, delay):

    car_conf = {"body_style" : "donkey", 
                "body_rgb" : (255, 0, 0),
                "car_name" : name,
                "font_size" : 25}

    # ~/projects/gym-donkeycar/gym_donkeycar/envs/donkey_env.py
    # generated_road, warehouse, sparkfun_avc, generated_track, mountain_track, roboracingleague_1, waveshare, mini_monaco, warren, thunderhill, circuit_launch
    scene_conf = {"msg_type" : "load_scene", "scene_name" : "generated_track"}

    cam_conf = {"msg_type" : "cam_config",
               "fov" : 0,
               "fish_eye_x" : 0,
               "fish_eye_y" : 0,
               "img_w" : 0,
               "img_h" : 0,
               "img_d" : 0,
               "img_enc" : 0,
               "offset_x" : 0,
               "offset_y" : 0,
               "offset_z" : 0,
               "rot_x" : 0,
               "rot_y" : 0,
               "rot_z" : 0}

    # Create client
    PORT = 9091
    SOCKET_READ_HZ = 1000 # read socket hz
    client = RaceClient(host=host, port=PORT, model_path=model_path, delay=delay, car_conf=car_conf, scene_conf=scene_conf, cam_conf=cam_conf, socket_read_hz=SOCKET_READ_HZ, name=name)
    try:
        while True:
            client.run_model()
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    except:
        traceback.print_exc()
        print("racer error!:")
    finally:
        client.stop()

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

