import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
from PIL import Image

def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--DoG_save_path', default='./DoG_outputs/', help='path to input image')
    
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)
    DoG = Difference_of_Gaussian(args.threshold)
    keypoints = DoG.get_keypoints(img)


    # 讀圖（要用彩色才能畫紅點）
    img_color = cv2.imread(args.image_path)

    # 畫 keypoints
    for y, x in keypoints:   # 注意你的格式是 (row, col)
        cv2.circle(img_color, (int(x), int(y)), 4, (255, 0, 0), -1)  # 藍色點

    im = Image.fromarray(img_color)
    im.save("output.png")

if __name__ == '__main__':
    main()