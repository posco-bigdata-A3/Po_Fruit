import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import socket
import pickle
import struct
import numpy as np
import cv2
import time
import threading
from PIL import Image
from trigger import TriggerClient

WS_PC = [0, 180, 0, 300]

def get_workspace_crop(img):
    retval = img[WS_PC[0]:WS_PC[1], WS_PC[2]:WS_PC[3], ...]
    return retval

class RealSenseClient:
    def __init__(self, host, port, fielt_bg=False):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.fielt_bg = fielt_bg
        self.buffer = b""
        self.remainingBytes = 0
        self.frame_length = None
        self.timestamp = None
        self.event = threading.Event()
        print(f"Connected to server at {self.host}:{self.port}")

    def get_camera_data(self, fielt_bg=None):
        if self.remainingBytes == 0:
            try:
                print("Sending request for frame...")
                self.socket.sendall(b"REQUEST_FRAME")
                
                print("Waiting for frame length...")
                header = self.socket.recv(12)
                if len(header) < 12:
                    print("Failed to receive complete header")
                    return None, None
                
                self.frame_length, self.timestamp = struct.unpack('<Id', header)
                self.remainingBytes = self.frame_length
                print(f"Receiving frame of length {self.frame_length} with timestamp {self.timestamp}")
            except Exception as e:
                print(f"Error receiving frame length: {e}")
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
                    fielt_bg = self.fielt_bg
                if fielt_bg:
                    mask = (cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)[:, :, 2] > 150)
                    color_img = color_img * mask[:, :, np.newaxis] + (1 - mask[:, :, np.newaxis]) * np.array([90, 89, 89])
                    color_img = color_img.astype(np.uint8)

                return depth_img, color_img
            else:
                return None, None
        except Exception as e:
            print(f"Error receiving frame data: {e}")
            return None, None

    def wait_for_trigger(self):
        self.event.wait()  # 이벤트가 설정될 때까지 대기
        self.event.clear()  # 이벤트 상태를 초기화
        return self.get_camera_data()

    def trigger(self):
        self.event.set()  # 이벤트를 설정하여 대기 중인 스레드를 깨움

def client_main(ip, port):
    global realsense_client
    realsense_client = RealSenseClient(ip, port, fielt_bg=True)
    
    counter = 0
    limit = 10
    sleep = 0.05

    all_rgbs = []
    while counter < limit:
        depth_img, color_img = realsense_client.wait_for_trigger()  # 한 번에 한 프레임만 요청
        if depth_img is None or color_img is None:
            print("Failed to get frames. Retrying...")
            time.sleep(sleep)
            continue
        im = Image.fromarray(color_img)
        
        print("depth img shape: ", depth_img.shape)
        print("color img shape: ", color_img.shape)
        counter += 1
        time.sleep(sleep)
        print('Step counter at {}'.format(counter))
        all_rgbs.append(color_img)

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))  # 오타 수정
        ax[0][0].imshow(color_img)
        ax[0][1].imshow(depth_img, cmap='gray')
        ax[1][0].imshow(get_workspace_crop(color_img))
        ax[1][1].imshow(get_workspace_crop(depth_img), cmap='gray')
        plt.show()

    print("Program finished.")

def trigger_listener(ip, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen(1)
    print(f"Trigger listener started on {ip}:{port}")

    conn, addr = server_socket.accept()
    print(f"Accepted trigger connection from {addr}")

    while True:
        data = conn.recv(1024)
        if data == b"TRIGGER":
            realsense_client.trigger()
            print("Received trigger")

if __name__ == '__main__':
    ip = "141.223.140.15"
    port = 1024
    trigger_port = 1025

    # 별도의 스레드에서 클라이언트 실행
    client_thread = threading.Thread(target=client_main, args=(ip, port))
    client_thread.start()

    # 별도의 스레드에서 트리거 리스너 실행
    trigger_listener_thread = threading.Thread(target=trigger_listener, args=(ip, trigger_port))
    trigger_listener_thread.start()

    trigger_client = TriggerClient(ip, trigger_port)
    while True:
        input("Press Enter to send trigger...")
        trigger_client.send_trigger()
