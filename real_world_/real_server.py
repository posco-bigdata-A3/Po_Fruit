import pyrealsense2 as rs
import socket
import pickle
import struct
import numpy as np
import threading

class RealSenseServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.pipeline = self.open_pipeline()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def open_pipeline(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        pipeline_profile = pipeline.start(cfg)
        return pipeline

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        timestamp = frames.get_timestamp()
        return depth_image, color_image, timestamp

    def handle_client(self, client_socket):
        while True:
            try:
                request = client_socket.recv(1024)
                if not request:
                    break
                print("Received request from client")
                with self.lock:
                    depth_image, color_image, timestamp = self.get_frames()
                    data = pickle.dumps((depth_image, color_image))
                    length = struct.pack('<I', len(data))
                    ts = struct.pack('<d', timestamp)
                    print(f"Sending frame of length {len(data)} with timestamp {timestamp}")
                    client_socket.sendall(length + ts + data)
                    print(f"Frame sent")
            except Exception as e:
                print(f"Error: {e}")
                break
        client_socket.close()

    def start(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Accepted connection from {addr}")
            client_handler = threading.Thread(target=self.handle_client, args=(client_socket,))
            client_handler.start()

if __name__ == '__main__':
    server = RealSenseServer('0.0.0.0', 1024)
    server.start()
