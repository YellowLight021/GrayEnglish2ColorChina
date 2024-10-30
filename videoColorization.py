import sys
from pathlib import Path
import PIL
import openvino as ov
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ColorizationPipeline:
    def __init__(
        self,
        model_path="ddcolor.xml",
        device="GPU"
    ):
        core = ov.Core()
        self.model = core.compile_model(model_path, device)

    def process(self, img):
        # Preprocess input image
        height, width = img.shape[:2]

        # Normalize to [0, 1] range
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)

        # Resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (512, 512))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1
        )
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)

        # Transpose HWC -> CHW and add batch dimension
        tensor_gray_rgb = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
        )

        # Run model inference
        output_ab = self.model(tensor_gray_rgb)[0]

        # Postprocess result
        # resize ab -> concat original l -> rgb
        output_ab_resize = (
            F.interpolate(torch.from_numpy(output_ab), size=(height, width))[0]
            .float()
            .numpy()
            .transpose(1, 2, 0)
        )

        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
        output_img = (output_bgr * 255.0).round().astype(np.uint8)

        return output_img

    def colorize_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            colorized_frame = self.process(frame)
            out.write(colorized_frame)

        cap.release()
        out.release()