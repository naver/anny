# Anny
# Copyright (C) 2025 NAVER Corp.
# Apache License, Version 2.0
import anny
import torch

dtype = torch.float32
device = torch.device("cpu")
model = anny.Anny().to(device=device, dtype=dtype)


# Typical COCO 23 ordering
keypoint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    ]
keypoints_regressor = anny.KeypointsRegressor(model, keypoint_labels, path="coco")

# Example of use
output = model()
keypoint_locations = keypoints_regressor(output)
print(keypoint_locations)
