# intel realsense카메라로 depth RGB데이터 불러오기
# (원래는 pykinect로 카메라 데이터 불러옴)
# pyrealsense2 라이브러리 설치

import pyrealsense2 as rs #Intel RealSense 카메라와 상호작용하기 위한 라이브러리
import pickle  
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image  #이미지 처리를 위한 라이브러리




class RealSenseClient:
    def __init__(self):
        self.pipeline = rs.pipeline() #RealSense 파이프라인을 생성
        self.config = rs.config() #파이프라인 설정을 위한 객체를 생성
        #RGB 스트림을 활성화
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #깊이 스트림을 활성화
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #설정된 스트림을 시작
        self.pipeline.start(self.config)
    
    #카메라로부터 RGB 및 깊이 데이터를 가져옵니다
    def get_camera_data(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        #컬러, 깊이 이미지를 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        depth_image = depth_image * 0.001  # Scale depth to meters

        return color_image, depth_image
    
    def stop(self):
        self.pipeline.stop()

if __name__ == '__main__':
    realsense = RealSenseClient()
    
    counter = 0 #현재 프레임 수를 저장하는 변수
    limit = 10 #수집할 최대 프레임 수를 설정
    sleep = 0.5 #각 프레임 사이의 지연 시간(초)

    all_rgbs = [] #수집된 모든 RGB 이미지를 저장하는 리스트
    
    #데이터 수집 루프
    try:
        while counter < limit:
            img, depth = realsense.get_camera_data() # RGB와 깊이 데이터를 가져옵니다
            if img is None or depth is None:
                continue
            
            im = Image.fromarray(img) #img를 PIL 이미지 객체로 변환
            
            #이미지와 깊이 데이터의 형태(shape)를 출력
            print("img shape: ", img.shape)
            print("depth shape: ", depth.shape)
            
            #counter를 증가시키고 지연 시간을 적용
            counter += 1
            time.sleep(sleep)
            print('Step counter at {}'.format(counter))
            

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(im)
            ax[0].set_title('RGB Image')
            ax[1].imshow(depth, cmap='gray')
            ax[1].set_title('Depth Image')
            plt.show()
    finally: #finally 블록에서 카메라 파이프라인을 정지합니다. 이는 데이터 수집이 완료되거나 오류가 발생해도 반드시 실행됩니다.
        realsense.stop()
