import numpy as np
import cv2
from utils import solve_homography, warping

def back_transform(src, canvas, corners):
    h, w, ch = src.shape
    x = np.array([[0, 0],
                  [w, 0],
                  [w, h],
                  [0, h]
                  ])
    H = solve_homography(corners, x)
    
    return  warping(canvas, src, H, 0, canvas.shape[0], 0, canvas.shape[1], direction='b')



if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    

    # TODO: call solve_homography() & warping
    # src = np.ones((h, w, c))*225
    src = np.zeros((h, w, c))
    output3_1 = back_transform(src, secret1, corners1)
    

    src = np.zeros((h, w, c))
    # src = np.ones((h, w, c))*225

    output3_2 = back_transform(src, secret2, corners2)

    cv2.imwrite('output3_1.png', output3_1.astype(np.uint8))
    cv2.imwrite('output3_2.png', output3_2.astype(np.uint8))