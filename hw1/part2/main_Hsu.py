import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    weights = []

    with open(args.setting_path, "r") as f:
        lines = f.readlines()
    for line in lines[1:6]:   
        nums = list(map(float, line.strip().split(',')))
        weights.append(nums)
    last_line = lines[6].strip().split(',')

    sigma = [float(last_line[1]), float(last_line[3])]
    JBF = Joint_bilateral_filter(int(sigma[0]),sigma[1])
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)#.astype(np.uint8)
    gray_jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)#.astype(np.uint8)
    #error_gray = np.sum(np.abs(bf_out.astype('int32')-gray_jbf_out.astype('int32')))
    iter = 0

    os.makedirs('test_image2', exist_ok=True)

    gray_jbf_out_bgr = cv2.cvtColor(gray_jbf_out ,cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"test_image2/filtered_{iter}.png", gray_jbf_out_bgr)
    iter += 1
    for w in weights:
        guidance = np.dot(img_rgb, w)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, guidance)#.astype(np.uint8)
        error = np.sum(np.abs(bf_out.astype('int32')-jbf_out.astype('int32')))
        print('error of ', w, ':', error)
        jbf_out_bgr = cv2.cvtColor(jbf_out ,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"test_image2/filtered_{w}.png",jbf_out_bgr)
        cv2.imwrite(f"test_image2/gray_{w}.png",guidance.astype(np.uint8))
        iter += 1
    
    
if __name__ == '__main__':
    main()