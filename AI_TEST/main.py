import matplotlib.pyplot as plt
import time
from kinect import RealSenseClient
from real_env import RealEnv

# Define the cropping values
WS_CROP_X = (0, 300)
WS_CROP_Y = (0, 300)

import torch

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print('Using GPU, device name:', torch.cuda.get_device_name(0))
    device = torch.device('cuda')
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

def main():
    realsense = RealSenseClient()

    task = "example_task" 
    all_objects = [ "keyboard", "human", "mouse"]
    task_objects = ["keyboard", "human"]
    env = RealEnv(realsense, task, all_objects, task_objects, device)  # device 인자를 추가합니다.

    counter = 0
    limit = 20
    sleep_time = 0.05

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))

    try:
        while counter < limit:
            obs = env.get_obs(save=True)
            if obs is None:
                continue
            print(f"Step counter at {counter}")
            print(f"color_im shape: {obs.color_im.shape}")
            print(f"depth_im shape: {obs.depth_im.shape}")

            ax[0][0].imshow(obs.color_im)
            ax[0][1].imshow(obs.depth_im, cmap='gray')
            ax[1][0].imshow(obs.color_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]])
            ax[1][1].imshow(obs.depth_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]], cmap='gray')

            # Call plot_preds to draw bounding boxes and display depth
            env.plot_preds(obs.color_im, obs.depth_im, obs.objects, save=False, show=False)

            plt.pause(0.001)
            counter += 1
            time.sleep(sleep_time)
    finally:
        realsense.stop()

if __name__ == "__main__":
    main()










