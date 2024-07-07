import pyrealsense2 as rs
import socket
import pickle
import struct
import numpy as np

def get_frames_and_timestamp(pipeline, depth_filter):
    frames = pipeline.wait_for_frames()
    frames.keep()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    if depth and color:
        depth2 = depth_filter.process(depth)
        depth2.keep()
        depth_data = depth2.as_frame().get_data()
        depth_mat = np.asanyarray(depth_data)
        color_data = color.get_data()
        color_mat = np.asanyarray(color_data)
        ts = frames.get_timestamp()
        return depth_mat, color_mat, ts
    else:
        return None, None, None

def open_pipeline():
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    pipeline_profile = pipeline.start(cfg)
    sensor = pipeline_profile.get_device().first_depth_sensor()
    return pipeline

def start_server(host, port):
    pipeline = open_pipeline()
    depth_filter = rs.decimation_filter()
    depth_filter.set_option(rs.option.filter_magnitude, 2)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Accepted connection from {addr}")

    try:
        while True:
            depth, color, timestamp = get_frames_and_timestamp(pipeline, depth_filter)
            if depth is not None and color is not None:
                data = pickle.dumps((depth, color))
                length = struct.pack('<I', len(data))
                ts = struct.pack('<d', timestamp)
                frame_data = length + ts + data

                conn.sendall(frame_data)
                print("Frame sent")
    finally:
        conn.close()
        server_socket.close()
        pipeline.stop()

if __name__ == '__main__':
    start_server('0.0.0.0', 1024)
