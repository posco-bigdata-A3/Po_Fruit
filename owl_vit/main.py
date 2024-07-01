import matplotlib.pyplot as plt
import time
from kinect import RealSenseClient
from real_env import RealEnv

# Define the cropping values
WS_CROP_X = (0, 300)
WS_CROP_Y = (0, 300)

def main():
    realsense = RealSenseClient()

    task = "example_task" 
    all_objects = ["biggest fruit", "small fruit_1","small fruit_2" ,"small fruit_3","small fruit_4"]
    task_objects = ["biggest fruit", "small fruit"]
    env = RealEnv(realsense, task, all_objects, task_objects)

    counter = 0
    limit = 20
    sleep_time = 0.05

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))

    try:
        while counter < limit:
            obs = env.get_obs(save=True)
            print(f"Step counter at {counter}")
            print(f"color_im shape: {obs.color_im.shape}")
            print(f"depth_im shape: {obs.depth_im.shape}")

            ax[0][0].imshow(obs.color_im)
            ax[0][1].imshow(obs.depth_im, cmap='gray')
            ax[1][0].imshow(obs.color_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]])
            ax[1][1].imshow(obs.depth_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]], cmap='gray')

            # Call plot_preds to draw bounding boxes
            env.plot_preds(obs.color_im, obs.objects, save=False, show=False)

            plt.pause(0.001)
            counter += 1
            time.sleep(sleep_time)
    finally:
        realsense.stop()

if __name__ == "__main__":
    main()


