"""
Donkey Car Simulator Benchmark
Usage:
    donkey-racer-bench.py (--host=<ip_address>) (--name=<car_name>)
    
Options:
    -h --help        Show this screen.
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)  # NOTSET:0, DEBUG:10, INFO:20, WARNING:30, ERROR:40, CRITICAL:50

class RaceClient:
    def __init__(self, host, port, car_conf, scene_conf, cam_conf, socket_read_hz=20):
        self.msg = None
        self.poll_socket_sleep_sec = 1.0/socket_read_hz
        self.car_conf = car_conf
        self.scene_conf = scene_conf
        self.cam_conf = cam_conf

        # the aborted flag will be set when we have detected a problem with the socket
        # that we can't recover from.
        self.aborted = False

        ### debug
        self.last_image_received_time = None
        self.current_image_received_time = None
        self.ok_frame_counter = 0
        self.ng_frame_counter = 0
        self.last_fps_time = None

        self.recv_queue = Queue.Queue(maxsize=10)
        self.send_queue = Queue.Queue(maxsize=10)
        self.image_queue = Queue.Queue(maxsize=10)

        # connect to unity simulator
        self.connect(host, port)

    def connect(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # connecting to the server 
        print(f'connecting to {host}:{port}')
        self.sock.connect((host, port))
        time.sleep(0.1)
        self.is_socket_read = True

    def send_now(self, msg):
        print("sending now:", msg)
        self.sock.sendall(msg.encode("utf-8"))

    def stop(self):
        self.sock.close()
        while not self.image_queue.empty():
            q = self.image_queue.get(block=False)
            self.image_queue.task_done()
        while not self.send_queue.empty():
            q = self.send_queue.get(block=False)
            self.send_queue.task_done()
        while not self.recv_queue.empty():
            q = self.recv_queue.get(block=False)
            self.recv_queue.task_done()


    def proc_msg(self):
        '''
        This is the thread message loop to process messages.
        We will send any message that is queued via the self.msg variable
        when our socket is in a writable state. 
        And we will read any messages when it's in a readable state and then
        call self.on_msg_recv with the json object message.
        '''
        self.sock.setblocking(False)
        inputs = [ self.sock ]
        outputs = [ self.sock ]
        partial = []

        while self.is_socket_read:
            # without this sleep, I was getting very consistent socket errors
            # on Windows. Perhaps we don't need this sleep on other platforms.
            time.sleep(self.poll_socket_sleep_sec)

            if True: #try:
                # test our socket for readable, writable states.
                readable, writable, exceptional = select.select(inputs, outputs, inputs)

                for sock in readable:
                    # print("waiting to recv")
                    try:
                        data = sock.recv(1024 * 64)
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
                            #m = replace_float_notation(m)
                            j = json.loads(m)
                            self.on_msg_recv(j)
                        else:
                            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                            print(m)
                            partial.append(m)
                            if last_char == '}':
                                if partial[0][0] == "{":
                                    assembled_packet = "".join(partial)
                                    #assembled_packet = replace_float_notation(assembled_packet)
                                    j = json.loads(assembled_packet)
                                    self.on_msg_recv(j)
                                else:
                                    print("failed packet.")
                                partial.clear()
                        
                for sock in writable:
                    while not self.send_queue.empty():
                        q = self.send_queue.get(block=False)
                        print("sending", q)
                        sock.sendall(q.encode("utf-8"))
                        self.send_queue.task_done()
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

        if msg['msg_type'] == "car_loaded":
            logger.debug("car_loaded")
            self.car_loaded = True

            self.cam_config_to_send_queue()
            self.car_config_to_send_queue()
        
        if msg['msg_type'] == "telemetry":
            imgString = msg["image"]
            ### to RGB
            image = Image.open(BytesIO(base64.b64decode(imgString)))
            image = np.asarray(image)

            ### interval counter
            self.current_image_received_time = time.time()
            if self.last_image_received_time is None:
                print(f'receive image: {self.current_image_received_time:10.7f}')
                self.last_fps_time = time.time()
            else:
                interval = self.current_image_received_time - self.last_image_received_time
                if interval <= 0.03 or interval >= 0.07:
                    print(f'receive image: {self.current_image_received_time:10.7f}, interval: {interval:.18f} - NG')
                    self.ng_frame_counter += 1
                else:
                    self.ok_frame_counter += 1
                    print(f'receive image: {self.current_image_received_time:10.7f}, interval: {interval:.18f}')
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
            steering = 0.0
            throttle = 0.0
            self.controls_to_send_queue(steering, throttle)
            #self.image_queue.put(image)

    def scene_config_to_send_queue(self, conf=None):
        logger.debug("scene_config_to_send_queue")
        if conf is None:
            conf = self.scene_conf
        msg = json.dumps(conf)
        self.send_queue.put(msg)

    def car_config_to_send_queue(self, conf=None):
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
            self.send_queue.put(msg)

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
        self.send_queue.put(msg)

    def controls_to_send_queue(self, steering, throttle):
        logger.debug("controls_to_send_queue: {steering}, {throttle}")
        p = {"msg_type" : "control",
             "steering" : steering.__str__(),
             "throttle" : throttle.__str__(),
             "brake" : "0.0"}
        msg = json.dumps(p)
        self.send_queue.put(msg)


def main(host, name):

    car_conf = {"body_style" : "donkey", 
                "body_rgb" : (255, 0, 0),
                "car_name" : name,
                "font_size" : 75}

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
    client = RaceClient(host=host, port=PORT, car_conf=car_conf, scene_conf=scene_conf, cam_conf=cam_conf, socket_read_hz=SOCKET_READ_HZ)
    try:
        client.proc_msg()
    except KeyboardInterrupt:
        pass
    except:
        import traceback
        traceback.print_exc()
    finally:
        client.stop()

if __name__ == '__main__':
    args = docopt(__doc__)
    main(host=args['--host'], name=args['--name'])

