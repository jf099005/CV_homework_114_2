import numpy as np
import cv2
from DoG import Difference_of_Gaussian

img = cv2.imread('testdata/1.png', 0).astype(np.float64)
img.shape
DoG = Difference_of_Gaussian(5.0)

# find keypoint from DoG and sort it
save_path = 'DoG_outputs'
import os
os.makedirs(save_path, exist_ok=True)
keypoints = DoG.get_keypoints(img, save_DoG_path=save_path)

# read GT
keypoints_gt = np.load('./testdata/1_gt.npy')
redundant = []
notfound = []
# Convert to sets of tuples for reliable comparison
set_yours = set(tuple(p) for p in keypoints)
set_gt = set(tuple(p) for p in keypoints_gt)

matches = set_yours.intersection(set_gt)
redundant = set_yours - set_gt
not_found = set_gt - set_yours

print(f"Total Yours: {len(set_yours)}")
print(f"Total GT: {len(set_gt)}")
print(f"Matches: {len(matches)}")
print(f"Actually Redundant: {len(redundant)}")
print(f"Actually Not Found: {len(not_found)}")