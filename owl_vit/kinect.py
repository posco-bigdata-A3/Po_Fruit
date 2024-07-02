import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

WS_PC = [180, 360, 300, 620]

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

    def get_camera_data(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_image = depth_image * 0.001  # Scale depth to meters

        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()


if __name__ == '__main__':
    realsense = RealSenseClient()
    
    counter = 0
    limit = 20
    sleep = 0.05
    reset_interval = 16  # 스트림을 재설정하는 간격

    all_rgbs = []
    try:
        while counter < limit:
            if counter % reset_interval == 0 and counter != 0:
                realsense.stop()
                time.sleep(1)  # 스트림이 완전히 종료되도록 잠시 대기
                realsense = RealSenseClient()
            
            img, depth = realsense.get_camera_data()
            if img is None or depth is None:
                continue
            
            im = Image.fromarray(img)
            
            print("img shape: ", img.shape)
            print("depth shape: ", depth.shape)
            counter += 1
            time.sleep(sleep)
            print('Step counter at {}'.format(counter))
            all_rgbs.append(img)

            fig, ax = plt.subplots(2, 2, figsize=(10, 5))
            ax[0][0].imshow(img)
            ax[0][1].imshow(depth, cmap='gray')
            ax[1][0].imshow(get_workspace_crop(img))
            ax[1][1].imshow(get_workspace_crop(depth), cmap='gray')
            plt.show()
    finally:
        realsense.stop()




