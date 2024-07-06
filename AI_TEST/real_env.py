import os
import dataclasses
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

WS_CROP_X = (0, 300)
WS_CROP_Y = (0, 300)

@dataclass
class EnvState:
    color_im: np.ndarray
    depth_im: np.ndarray
    objects: Dict[str, torch.Tensor]  # GPU 텐서를 사용하도록 수정

class RealEnv():
    def __init__(self, bin_cam, task: str, all_objects: List[str], task_objects: List[str], device: torch.device, output_name: str = None):
        self.bin_cam = bin_cam
        self.task = task
        self.all_objects = all_objects
        self.task_objects = task_objects
        self.device = device # 추가된 부분
        if output_name is None:
            self.output_name = f"real_world/outputs/{self.task}/"
        else:
            self.output_name = f"real_world/outputs/{output_name}/"
        os.makedirs(self.output_name, exist_ok=True)

        self.robot_name = 'Bob'
        self.human_name = 'Alice'

        # Load OWL-ViT model
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device) # 모델을 GPU로 이동
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

        self.timestep = 0

    def get_obs(self, save=False) -> EnvState:
        # Get color and depth images from the kinect
        color_im, depth_im = self.bin_cam.get_camera_data()
        ws_color_im = color_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]]
        ws_depth_im = depth_im[WS_CROP_X[0]:WS_CROP_X[1], WS_CROP_Y[0]:WS_CROP_Y[1]]

        image = Image.fromarray(ws_color_im)
        text = self.all_objects

        # Get max probability bounding boxes for each object label
        inputs = self.processor(text=[text], images=image, return_tensors="pt").to(self.device) # 입력을 GPU로 이동
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device) # 타겟 사이즈를 GPU로 이동
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        pred_scores = results[0]["scores"].detach() # GPU에서 결과 유지
        pred_labels, pred_boxes = results[0]["labels"].detach(), results[0]["boxes"].detach() # GPU에서 결과 유지

        objects = {}
        for label in torch.unique(pred_labels):
            max_score_idx = torch.argmax(pred_scores[pred_labels == label])
            max_box = pred_boxes[pred_labels == label][max_score_idx]
            objects[text[label]] = max_box # GPU 텐서를 그대로 사용

        self.timestep += 1
        if save:
            image.save(f"{self.output_name}/img_{self.timestep}.png")
        obs = EnvState(
            color_im=color_im,
            depth_im=depth_im,
            objects=objects,
        )
        return obs

    def plot_preds(self, color_im, objects, save=False, show=True):
        fig, ax = plt.subplots(figsize=(12, 12 * color_im.shape[0] / color_im.shape[1]))
        ax.imshow(color_im)
        colors = sns.color_palette('muted', len(objects))
        for label, c in zip(objects, colors):
            (xmin, ymin, xmax, ymax) = objects[label].cpu().numpy() # numpy 배열로 변환하여 사용
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
            if label in self.task_objects:
                ax.text(xmin-30, ymax+15, label, fontsize=22, bbox=dict(facecolor='white', alpha=0.8))
            else:
                ax.text(xmin, ymin-10, label, fontsize=22, bbox=dict(facecolor='white', alpha=0.8))
        plt.axis('off')
        fig.tight_layout()

        if show:
            plt.show()
        if save:
            fig.savefig(f"{self.output_name}/pred_{self.timestep}.png")
















