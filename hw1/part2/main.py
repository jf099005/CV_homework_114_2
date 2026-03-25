import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
import csv

def read_file(file_path):
    colors = []
    sigma = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Process lines
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # If it's the header or RGB data
        if line[0].isdigit() or line.startswith("R"):
            if line.startswith("R"):
                continue  # skip header
            r, g, b = map(float, line.split(","))
            colors.append((r, g, b))
        
        # If it's the parameter line
        elif "sigma" in line:
            parts = line.split(",")
            for i in range(0, len(parts), 2):
                key = parts[i]
                value = float(parts[i + 1])
                # params[key] = value
                sigma.append(value)
    return colors, sigma

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    colors, sigma1, sigma2 = read_file(args.setting_path)

    
    for (r,g,b) in colors:
        img_gray = r*img_rgb[:,:,0] + g*img_rgb[:,:,1] + b*img_rgb[:,:,2]
        os.makedirs(f'./outputs/r{r}_g{g}_b{b}', exist_ok=True)
        


    ### TODO ###


if __name__ == '__main__':
    main()