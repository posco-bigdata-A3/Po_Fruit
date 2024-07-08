import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import socket
import pickle
import struct
import numpy as np
import cv2
import threading
from PIL import Image

class RealSenseClient:
    def __init__(self, host, port, fielt_bg=False):
        self.host = host
        self.port = port
        self.fielt_bg = fielt_bg
        self.buffer = b""
        self.remainingBytes = 0
        self.frame_length = None
        self.timestamp = None
        self.socket = None
        self.connect_to_server()

    @property
    def color_intr(self):
        # 반환되는 내부 파라미터는 numpy 배열로 변환되어야 합니다.
        return np.array([[self.intr.fx, 0, self.intr.ppx],
                         [0, self.intr.fy, self.intr.ppy],
                         [0, 0, 1]])

    def connect_to_server(self):
        if self.socket:
            self.socket.close()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
        except ConnectionRefusedError as e:
            print(f"Connection failed: {e}")
            self.socket = None

    def get_camera_data(self, fielt_bg=None):
        if not self.socket:
            self.connect_to_server()
            print("Connect Complete")
            if not self.socket:
                return None, None

        if self.remainingBytes == 0:
            try:
                print("Sending trigger...")
                self.socket.sendall(b"TRIGGER")
                
                print("Waiting for frame length...")
                header = self.socket.recv(12)
                if len(header) < 12:
                    print("Failed to receive complete header")
                    return None, None
                
                self.frame_length, self.timestamp = struct.unpack('<Id', header)
                self.remainingBytes = self.frame_length
                print(f"Receiving frame of length {self.frame_length} with timestamp {self.timestamp}")
            
            except Exception as e:
                #print(f"Erprint(f"Connection failed: {e}")ror receiving frame length: {e}")
                return None, None
        
        try:
            while self.remainingBytes > 0:
                data = self.socket.recv(self.remainingBytes)
                if not data:
                    print("Failed to receive data chunk")
                    return None, None

                self.buffer += data
                self.remainingBytes -= len(data)
            
            if len(self.buffer) == self.frame_length:
                print("Received complete frame")
                depth_img, color_img = pickle.loads(self.buffer)
                self.buffer = b""
                self.remainingBytes = 0

                # Apply depth offset
                depth_img = depth_img.astype(np.float64) * 0.973
                depth_img = depth_img.astype(np.uint16)

                # Apply background filter if enabled
                if fielt_bg is None:
                    fielt_bg =5 hours ago self.fielt_bg

                return depth_img, color_img
            else:
                return None, None
        except Exception as e:
            print(f"Error receiving frame data: {e}")
            return None, None

    def send_trigger(self):
        return self.get_camera_data()

# 클라이언트를 실행하는 별도의 스크립트
# if __name__ == '__main__':
#     #ip = "141.223.140.15"
#     ip = '192.168.0.6'
#     port = 1024

#     # RealSenseClient 인스턴스 생성
#     realsense_client = RealSenseClient(ip, port, fielt_bg=True)

#     while True:
#         input("Press Enter to send trigger...")
#         depth_img, color_img = realsense_client.send_trigger()
#         if depth_img is not None and color_img is not None:
#             print("Depth image shape:", depth_img.shape)
#             print("Color image shape:", color_img.shape)
#         else:
#             print("Failed to receive images.")


#    def color_intr(self):
        # 반환되는 내부 파라미터는 numpy 배열로 변환되어야 합니다.
#        return np.array([[self.intr.fx, 0, self.intr.ppx],
#                         [0, self.intr.fy, self.intr.ppy],
#                         [0, 0, 1]])
