"""Some functions useful for running Unity within Python."""
import numpy as np
import random
import socket
import struct
import subprocess
import time


class TCPUnityParser(object):
    """Helper class for listening/parsing messages from Unity.

    The various functions, titled 'parse_VAR', are all designed to easily
    listen for and parse the data of interest.
    """
    def __init__(self, unity_socket):
        self.unity_socket = unity_socket
        self.unity_socket.listen(4)
        connection, _ = self.unity_socket.accept()
        time.sleep(0.1)
        self.connection = connection

    def close(self):
        """Close all open sockets/connections."""
        self.connection.close()
        self.unity_socket.close()

    def parse_filler(self, filler_len):
        """Listen for a set number of bytes (no return value)."""
        self.connection.recv(filler_len)

    def parse_struct(self, data_type):
        """Parse a single value of a set data_type.

        See the documentation for the 'struct' module for a list of data types.
        """

        if data_type == 'i':
            return struct.unpack('i', self.connection.recv(4))[0]
        elif data_type == 'f':
            return struct.unpack('f', self.connection.recv(4))[0]
        elif data_type == 'q':
            return struct.unpack('q', self.connection.recv(8))[0]
        else:
            raise ValueError("Unknown data_type '{}'".format(data_type))

    def parse_string(self):
        """Listen for a string (terminated in '\x00')."""
        data_name = ''
        stop_char = b'\x00'[0]
        while True:
            letter = self.connection.recv(1)[0]
            if letter == stop_char:
                break
            data_name += chr(letter)

        return data_name

    def parse_image(self):
        """Listens for Unity image data.

        Unity provides more data than we care to use, yet the data is
        stored in 'data' (a dictionary) in case it is ever wanted.
        """

        self.parse_filler(4)

        data = dict()
        data['type'] = self.parse_struct('i')
        data['tick'] = self.parse_struct('q')
        data['name'] = self.parse_string()

        data['im_fov'] = self.parse_struct('f')
        data['im_clip'] = self.parse_struct('f')
        data['im_height'] = self.parse_struct('i')
        data['im_width'] = self.parse_struct('i')

        im_data = b''
        im_data_target_len = data['im_height'] * data['im_width'] * 3
        while im_data_target_len - len(im_data) > 0:
            im_data += self.connection.recv(im_data_target_len - len(im_data))
        image = np.frombuffer(im_data, dtype=np.uint8)
        image = np.reshape(image, [data['im_width'], data['im_height'], 3])
        image = np.flip(image, axis=0)
        return image


class UnityBridge(object):
    def __init__(
            self,
            unity_exe,
            unity_args='-batchmode -screen-fullscreen 0 -logFile /data/unity_logs.txt',
            tcp_ip='127.0.0.1',
            talk_port=None,
            listen_port=None,
            is_debug=False,
            sim_scale=1.0):

        self.do_buffer = False
        self.messages = []
        self.tcp_ip = tcp_ip
        if talk_port:
            self.talk_port = talk_port
        else:
            s = socket.socket()
            s.bind(("", 0))
            self.talk_port = int(s.getsockname()[1])

        if listen_port:
            self.listen_port = listen_port
        else:
            s = socket.socket()
            s.bind(("", 0))
            self.listen_port = int(s.getsockname()[1])

        if is_debug:
            print(("Ports: py-listen={} py-talk={}".format(self.listen_port,
                                                           self.talk_port)))

        self.unity_exe = unity_exe
        self.unity_args = unity_args + ' -talk-port {} -listen-port {}'.format(
            self.listen_port, self.talk_port)

        self.is_debug = is_debug
        self.sim_scale = sim_scale

    def start_unity(self):
        """Start Unity as a background process.

        Before starting Unity, we open a tcp socket. When Unity opens, it
        connects to that socket, which we listen to via 'unity_listener'.
        Note that this code must be run in this order: (1) open socket, (2)
        start Unity, (3) listen to socket (via TCPUnityParser). In any other
        order Unity will not connect, or Python may hang.
        """

        # Launch unity as a background process
        if not self.is_debug:
            unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            unity_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            unity_socket.bind((self.tcp_ip, self.listen_port))
            self.unity_subprocess = subprocess.Popen(
                [self.unity_exe] + self.unity_args.split(" "),
                stdout=subprocess.DEVNULL)
        else:
            unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            unity_socket.bind((self.tcp_ip, self.listen_port))
            print(unity_socket)

        self.unity_listener = TCPUnityParser(unity_socket)

    def send_message(self, message, pause=0.1):
        if self.do_buffer:
            self.messages.append(message)
        else:
            self.talker.send(message.encode())
            if pause > 0:
                time.sleep(pause)

    def __enter__(self):
        self.start_unity()
        self.start_talker()
        return self

    def __exit__(self, type, value, traceback):
        self.do_buffer = False
        self.send_message("shutdown shutdown", pause=1.0)
        self.unity_listener.close()
        self.talker.close()
        # Wait for the unity process to finish
        if not self.is_debug:
            self.unity_subprocess.wait()
        time.sleep(1.0)

    def start_talker(self):
        self.talker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.talker.connect((self.tcp_ip, self.talk_port))

    def create_cube(self):
        rx = random.random() * 10
        ry = random.random() * 10
        rz = random.random() * 10
        self.send_message("main_builder cube {} {} {}".format(rx, ry, rz),
                          pause=0.001)

    def create_object(self, command_name, pose, height):
        self.send_message("main_builder {} {} {} {}".format(
            command_name,
            pose.x * self.sim_scale,
            pose.y * self.sim_scale, height), pause=0.001)

    def get_image(self, camera_name, pause=-1):
        self.send_message(camera_name + " render", pause)
        return self.unity_listener.parse_image()
