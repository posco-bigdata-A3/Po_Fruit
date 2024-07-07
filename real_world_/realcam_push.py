#!/usr/bin/python
import pyrealsense2 as rs
import sys
import asyncore
import numpy as np
import pickle
import socket
import struct
import cv2

class RealSenseServer(asyncore.dispatcher):
    def __init__(self, address):
        asyncore.dispatcher.__init__(self)
        self.pipeline = self.open_pipeline()
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bind(address)
        self.listen(5)
        print(f'Server listening on {address}')

    def open_pipeline(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline = rs.pipeline()
        pipeline.start(cfg)
        return pipeline

    def handle_accept(self):
        sock, addr = self.accept()
        print('Accepted connection from', addr)
        handler = ImageClient(sock, self.pipeline)

class ImageClient(asyncore.dispatcher_with_send):
    def __init__(self, sock, pipeline):
        asyncore.dispatcher_with_send.__init__(self, sock)
        self.pipeline = pipeline
        self.frame_data = None

    def handle_read(self):
        self.update_frame()
        self.send_frame()

    def update_frame(self):
        color_img, depth_img, timestamp = self.get_frames()
        if color_img is not None and depth_img is not None:
            data = pickle.dumps((color_img, depth_img))
            length = struct.pack('<I', len(data))
            ts = struct.pack('<d', timestamp)
            self.frame_data = length + ts + data
            print("Updated frame")

    def send_frame(self):
        if self.frame_data:
            try:
                sent = self.send(self.frame_data)
                print(f"Sent {sent} bytes")
                self.frame_data = self.frame_data[sent:]
            except Exception as e:
                print(f"Send failed: {e}")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None, None
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        timestamp = frames.get_timestamp()
        return color_img, depth_img, timestamp

def main(argv):
    server_address = ('0.0.0.0', 1024)
    server = RealSenseServer(server_address)
    asyncore.loop()

if __name__ == '__main__':
    main(sys.argv[1:])
