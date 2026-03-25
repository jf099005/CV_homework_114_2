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
    return np.sum( np.abs(img1 - img2) )#/(img1.shape[0]*img1.shape[1])

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    filename = Path(args.image_path).stem
    print('processing file', filename)

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    jbf_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_rgb).astype(np.uint8)
    jbf_filtered_bgr = cv2.cvtColor(jbf_filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_root, 'img_jbf.png'), jbf_filtered_bgr)

    colors, (sigma_s, sigma_r) = read_file(args.setting_path)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)

    print('Running Bilateral filter with sigma_s', sigma_s, ', sigma_r', sigma_r)
    output_root = f'output_{filename}'
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(args.image_path, f'{output_root}/ref_image.png')

    print('\t testing cv2-conversion')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    output_folder = f'{output_root}/cv2_transform'
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite( os.path.join(output_folder, 'img_gray.png'), img_gray )
    # jbf_gray_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_gray).astype(np.uint8)
    jbf_gray_filtered = JBF.joint_bilateral_filter(img = jbf_filtered, guidance = img_gray).astype(np.uint8)

    jbf_gray_filtered_bgr = cv2.cvtColor(jbf_gray_filtered, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, 'img_jbf.png'), jbf_gray_filtered_bgr)
    cv2.imwrite( os.path.join(output_folder, 'img_gray.png'), img_gray )
    error = L1Norm(jbf_filtered, jbf_gray_filtered)
    print("\t\t L1 Norm:", error)


    for (r,g,b) in colors:
        print('\t testing color (r, g, b)', (r,g,b))
        img_gray = (r*img_rgb[:,:,0] + g*img_rgb[:,:,1] + b*img_rgb[:,:,2])
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

        # print('shape of gray:', img_gray.shape)

        output_folder = f'{output_root}/r{r}_g{g}_b{b}'
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite( os.path.join(output_folder, 'img_gray.png'), img_gray )
        # jbf_gray_filtered = JBF.joint_bilateral_filter(img = img_rgb, guidance = img_gray).astype(np.uint8)
        jbf_gray_filtered = JBF.joint_bilateral_filter(img = jbf_filtered, guidance = img_gray).astype(np.uint8)
        error = L1Norm(jbf_filtered, jbf_gray_filtered)
        print("\t\t L1 Norm:", error)

        jbf_gray_filtered_bgr = cv2.cvtColor(jbf_gray_filtered, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_folder, 'img_jbf.png'), jbf_gray_filtered_bgr)
        # gray_out_bgr = cv2.cvtColor(img_gray, )
        cv2.imwrite( os.path.join(output_folder, 'img_gray.png'), img_gray )

    ### TODO ###


if __name__ == '__main__':
    main()