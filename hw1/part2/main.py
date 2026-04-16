import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
import csv
import shutil
from pathlib import Path
def read_file(file_path):
    colors = []
    # sigma = []
    sigma_s = None
    sigma_r = None

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
                if 'sigma_s' in key.lower():
                    sigma_s = int(value)
                elif 'sigma_r' in key.lower():
                    sigma_r = value
                # sigma.append(value)
    return colors, (sigma_s, sigma_r)

def L1Norm(img1, img2):
    assert img1.shape == img2.shape
    return np.sum( np.abs(img1.astype(np.int32) - img2.astype(np.int32)) )#/(img1.shape[0]*img1.shape[1])

def check_img(img_jbf, guidance, baseline, output_folder = None):
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(os.path.join(output_folder, 'img_jbf.png'), cv2.cvtColor(img_jbf, cv2.COLOR_RGB2BGR))
        cv2.imwrite( os.path.join(output_folder, 'img_gray.png'), guidance )

    error = L1Norm(baseline, img_jbf)
    print("\t\t L1 Norm:", error)
    return error



def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    filename = Path(args.image_path).stem
    print('processing file', filename)

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    colors, (sigma_s, sigma_r) = read_file(args.setting_path)
    print('Running Bilateral filter with sigma_s', sigma_s, ', sigma_r', sigma_r)
    output_root = f'output_{filename}'
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(args.image_path, f'{output_root}/ref_image.png')

    JBF = Joint_bilateral_filter(sigma_s = sigma_s, sigma_r = sigma_r)

    jbf_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_rgb).astype(np.uint8)
    baseline = jbf_filtered

    output_folder = os.path.join(output_root, 'COLOR_BGR2GRAY')
    print('\t testing cv2-conversion')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    jbf_gray_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_gray).astype(np.uint8)

    check_img(jbf_gray_filtered, img_gray, baseline, output_folder)

    for (r,g,b) in colors:
        print('\t testing color (r, g, b)', (r,g,b))
        img_gray = (r*img_rgb[:,:,0] + g*img_rgb[:,:,1] + b*img_rgb[:,:,2])
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)
        output_folder = f'{output_root}/r{r}_g{g}_b{b}'
        jbf_gray_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_gray).astype(np.uint8)    
        check_img(jbf_gray_filtered, img_gray, baseline, output_folder)


    # 在 main() 最後加上這段
    print('\n--- Sanity check ---')
    # 用全黑 guidance（全部對比度為 0），range kernel 全為 1，退化為純 Gaussian blur
    zero_guidance = np.zeros_like(img_gray)
    jbf_zero = JBF.joint_bilateral_filter(img=img_rgb, guidance=zero_guidance).astype(np.uint8)
    print('Zero guidance (pure Gaussian) L1:', L1Norm(img_rgb, jbf_zero))

    # 用原圖 guidance（完整 bilateral），L1 應該最小
    jbf_self = JBF.joint_bilateral_filter(img=img_rgb, guidance=img_rgb).astype(np.uint8)
    print('Self guidance (full bilateral) L1:', L1Norm(img_rgb, jbf_self))


if __name__ == '__main__':
    main()