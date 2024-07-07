import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import cv2  
from PIL import Image
import time

WS_PC = [0, 180, 0, 300]

def get_workspace_crop(img):
    retval = img[WS_PC[0]:WS_PC[1], WS_PC[2]:WS_PC[3], ...]
    return retval

class RealSenseClient:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)
    
    def get_camera_data(self, n=1):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        
        return color_img, depth_img

    def get_intrinsics(self):
        profile = self.pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return np.array([[intr.fx, 0, intr.ppx],
                         [0, intr.fy, intr.ppy],
                         [0, 0, 1]])

if __name__ == '__main__':
    
    realsense = RealSenseClient()
    
    counter = 0 #카운터를 0으로 초기화
    limit = 10 #반복 한계를 100으로 설정
    sleep = 0.05 #각 반복 사이의 대기 시간을 0.05초로 설정

    all_rgbs = [] #수집한 색상 이미지를 저장할 리스트를 초기화
    while counter < limit:
        img, depth = realsense.get_camera_data()
        if img is None or depth is None:
            print("Failed to get frames. Retrying...")
            continue
        
        im = Image.fromarray(img)
        
        print("img shape: ", img.shape)
        print("depth shape: ", depth.shape)
        counter += 1
        time.sleep(sleep) #각 반복 사이에 0.05초 동안 실행을 멈춥니다. 시스템 과부하를 방지
        print('Step counter at {}'.format(counter))
        all_rgbs.append(img)

        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        ax[0][0].imshow(img)
        ax[0][1].imshow(depth)
        ax[1][0].imshow(get_workspace_crop(img))
        ax[1][1].imshow(get_workspace_crop(depth))
        plt.show()

