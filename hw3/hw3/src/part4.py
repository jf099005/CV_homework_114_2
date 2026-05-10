import math
import random
import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def match_nodes(img1, img2, metric = cv2.NORM_HAMMING):
    """
    Align the features of two images by their coordinates
    :param features_img1: features of image 1
    :param features_img2: features of image 2
    :param metric: distance metric for matching
    :return: aligned feature pairs
    """

    feature_extractor = cv2.ORB_create()

    kp1, des1 = feature_extractor.detectAndCompute(img1, None)
    kp2, des2 = feature_extractor.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(metric, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # return matches #[ (kp1[m.trainIdx].pt, kp2[m.queryIdx].pt) for m in matches ]
    return [ kp1[m.queryIdx].pt for m in matches ], [kp2[m.trainIdx].pt for m in matches ]

    # print(matches)
    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # plt.imshow(img3),plt.show()

def RANSAC(node_matches, N_sample, n_iter, threshold, n_accept):
    H_opt = None
    n_inlier_opt = 0
    # node_matches = np.array(node_matches)
    # node_matches = np.concatenate(
    #     [node_matches, np.ones((node_matches.shape[0], node_matches.shape[1], 1))],
    #     axis=2
    # )

    for _ in range(n_iter):
        sampled =  random.sample(node_matches, N_sample)
        # sampled_nodes = np.array(sampled)
        # nodes_src = np.array([p[0] for p in sampled])
        nodes_src = np.array([p[0] for p in sampled])
        # print("src shape", nodes_src.shape)
        # nodes_src = np.concatenate(
        #     [nodes_src, np.ones((nodes_src.shape[0], 1))],
        #     axis=1
        # )
        nodes_dst = np.array([p[1] for p in sampled])
        # nodes_dst = np.concatenate(
        #     [nodes_dst, np.ones((nodes_dst.shape[0], 1))],
        #     axis=1
        # )

        H = solve_homography(nodes_src, nodes_dst)
        # print('H shape', H.shape)

        nodes_src = np.concatenate(
            [nodes_src, np.ones((nodes_src.shape[0], 1))],
            axis=1
        )


        # print('src shape', nodes_src.shape)

        mapping_dst_h = (H@nodes_src.T).T
        mapping_dst = mapping_dst_h[:, :2] / mapping_dst_h[:, 2:]

        diff = np.linalg.norm(nodes_dst - mapping_dst, axis=1)
        n_inlier = np.sum(diff < threshold)
        print('inlier:', n_inlier)
        print("avg dis:", diff.mean())
        if n_inlier >= n_accept:
            return H
        elif n_inlier > n_inlier_opt:
            H_opt = H
    return H_opt


def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    # out = None

    # for all images to be stitched:

    p = 0.95

    # N_sample = int(math.log(1-p)/math.log(1-(1-e)))

    for idx in tqdm(range(len(imgs)-1)):
        print("============idx {}==============".format(idx))
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        matches = match_nodes(im1, im2)
        n_matches = len(matches)
        H = RANSAC(
            matches,
            N_sample = 4,#n_matches//2,
            n_iter = 10,
            threshold = 10,
            n_accept = 10
        )
        print('dims', last_best_H.shape, H.shape)
        dst = warping(im2, dst, last_best_H@H, 0, im2.shape[0], 0, im2.shape[1], 'b')
        last_best_H = last_best_H@H
        # print("matches")
        # print(node_matches)
        # TODO: 1.feature detection & matching

        # TODO: 2. apply RANSAC to choose best H

        # TODO: 3. chain the homographies

        # TODO: 4. apply warping

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)