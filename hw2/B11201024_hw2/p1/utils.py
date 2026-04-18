# ============================================================================
# File: util.py
# Date: 2026-03-27
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
# import cv2
from cyvlfeat.sift.dsift import dsift
from sklearn.cluster import KMeans, MiniBatchKMeans

from scipy.spatial.distance import cdist
from statistics import mode
import os
CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: str, height: int=16, width: int=16):
    '''
    Build tiny image features.
    - Args: : 
        - img_paths (N): list of string of image paths
    - Returns: :
        - tiny_img_feats (N, d): ndarray of resized and then vectorized
                                 tiny images
    NOTE:
        1. N is the total number of images
        2. if the images are resized to 16x16, d would be 256
    '''
    
    #################################################################
    # TODO:                                                         #
    # To build a tiny image feature, you can follow below steps:    #
    #    1. simply resize the original image to a very small        #
    #       square resolution, e.g. 16x16. You can either resize    #
    #       the images to square while ignoring their aspect ratio  #
    #       or you can first crop the center square portion out of  #
    #       each image.                                             #
    #    2. flatten and normalize the resized image.                #
    #################################################################

    tiny_img_feats = []

    # if not os.path.exists(img_paths):
    #     return None
    # print(img_paths)
    # for img_name in os.listdir(img_paths):
    #     sample_path = os.path.join(img_paths, img_name)
    for sample_path in img_paths:
        # sample_path = os.path.join(img_paths, img_name)

        img = Image.open(sample_path)
        tiny_img = img.resize( (height, width) )
        tiny_img = np.array(tiny_img)
        tiny_img = tiny_img - np.mean(tiny_img)
        tiny_img = tiny_img / (np.linalg.norm(tiny_img) + 1e-8)

        tiny_img_feats.append( tiny_img.flatten() )
        # pass

    #################################################################
    #                        END OF YOUR CODE                       #
    #################################################################

    tiny_img_feats = np.array(tiny_img_feats)
    print(tiny_img_feats.shape)
    return tiny_img_feats

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400,
        debug = False, 
        stepsize = 5
    ):
    '''
    Args:
        img_paths (N): list of string of image paths (training)
        vocab_size: number of clusters desired
    Returns:
        vocab (vocab_size, sift_d): ndarray of clusters centers of k-means
    NOTE:
        1. sift_d is 128
        2. vocab_size is up to you, larger value will works better
           (to a point) but be slower to compute,
           you can set vocab_size in p1.py
    '''
    
    ##################################################################################
    # TODO:                                                                          #
    # To build vocabularies from training images, you can follow below steps:        #
    #   1. create one list to collect features                                       #
    #   2. for each loaded image, get its 128-dim SIFT features (descriptors)        #
    #      and append them to this list                                              #
    #   3. perform k-means clustering on these tens of thousands of SIFT features    #
    # The resulting centroids are now your visual word vocabulary                    #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful functions                                                          #
    #   Function : dsift(img, step=[x, x], fast=True)                                #
    #   Function : kmeans(feats, num_centers=vocab_size)                             #
    #                                                                                #
    # NOTE:                                                                          #
    # Some useful tips if it takes too long time                                     #
    #   1. you don't necessarily need to perform SIFT on all images, although it     #
    #      would be better to do so                                                  #
    #   2. you can randomly sample the descriptors from each image to save memory    #
    #      and speed up the clustering, which means you don't have to get as many    #
    #      SIFT features as you will in get_bags_of_sift(), because you're only      #
    #      trying to get a representative sample here                                #
    #   3. the default step size in dsift() is [1, 1], which works better but        #
    #      usually become very slow, you can use larger step size to speed up        #
    #      without sacrificing too much performance                                  #
    #   4. we recommend debugging with the 'fast' parameter in dsift(), this         #
    #      approximate version of SIFT is about 20 times faster to compute           #
    #   5. If you have trouble installing cyvlfeat, you can use cv2 for sift & 
    #      sklearns for kmeans
    # You are welcome to use your own SIFT feature                                   #
    ##################################################################################
    vocab_pool = []

    for img_path in tqdm(img_paths, desc = 'generating vocab...'):
        img = Image.open(img_path)
        img = np.array(img)
        _, descriptions = dsift(img, step = [stepsize, stepsize], fast = debug)

        l2_norms = np.linalg.norm(descriptions, axis=1, keepdims=True)

        # 加上 1e-8 是一個好習慣，避免遇到全為 0 的向量導致「除以零」的錯誤
        descriptions = descriptions / (l2_norms + 1e-8)

        vocab_pool.extend(descriptions.astype(np.float32))

        # for desp in descriptions:
        #     vocab_pool.append( desp.astype(np.float32) )

    vocab_pool = np.array(vocab_pool)
    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    
    
    # from sklearn.cluster import KMeans

    kmeans_model = MiniBatchKMeans(n_clusters=vocab_size, batch_size=1000)
    
    # kmeans_model = KMeans(
    #     n_clusters=vocab_size,
    #     max_iter=100,
    #     verbose=1,
    #     n_init=1
    # )

    kmeans_model.fit(vocab_pool)
    print('End of build vocabulary')
    vocab = kmeans_model.cluster_centers_

    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array,
        stepsize = 5,
        debug = False
    ):
    '''
    Args:
        img_paths (N): list of string of image paths
        vocab (vocab_size, sift_d) : ndarray of clusters centers of k-means
    Returns:
        img_feats (N, d): ndarray of feature of images, each row represent
                          a feature of an image, which is a normalized histogram
                          of vocabularies (cluster centers) on this image
    NOTE :
        1. d is vocab_size here
    '''

    ############################################################################
    # TODO:                                                                    #
    # To get bag of SIFT words (centroids) of each image, you can follow below #
    # steps:                                                                   #
    #   1. for each loaded image, get its 128-dim SIFT features (descriptors)  #
    #      in the same way you did in build_vocabulary()                       #
    #   2. calculate the distances between these features and cluster centers  #
    #   3. assign each local feature to its nearest cluster center             #
    #   4. build a histogram indicating how many times each cluster presents   #
    #   5. normalize the histogram by number of features, since each image     #
    #      may be different                                                    #
    # These histograms are now the bag-of-sift feature of images               #
    #                                                                          #
    # NOTE:                                                                    #
    # Some useful functions                                                    #
    #   Function : dsift(img, step=[x, x], fast=True)                          #
    #   Function : cdist(feats, vocab)                                         #
    #                                                                          #
    # NOTE:                                                                    #
    #   1. we recommend first completing function 'build_vocabulary()'         #
    ############################################################################

    img_feats = []

    ############################################################################
    #                                END OF YOUR CODE                          #
    ############################################################################
        #print("step_sample", step_sample)
    image_feats = []
    i=-1
    # sift = cv2.SIFT_create()

    vocab_size = vocab.shape[0]
    for path in tqdm(img_paths, desc='get bag of words:'):
        i+=1
        #if (not i%20):
        #    print(" i = ", i)

        img = Image.open(path)
        img = np.array(img)
        frames, descriptors = dsift(img, step=[stepsize, stepsize], fast=debug)

        # img = Image.open(path).convert('L')
        # img = np.array(img)

        # keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            # 沒特徵 → 給全 0 histogram
            hist = np.zeros(vocab_size)
            image_feats.append(hist)
            continue

        #histogram
        #for each img:
        #    for each feature: (may different)
        #        find closet feature from vocab 
        dist = cdist(descriptors, vocab)  
        # kmin = np.argmin(dist, axis = 0)
        nearest_words = np.argmin(dist, axis=1)
        # hist, bin_edges = np.histogram(kmin, bins=len(vocab))
        hist, _ = np.histogram(nearest_words, bins=np.arange(vocab_size + 1))
        # hist_norm = [float(i)/sum(hist) for i in hist]

        hist = hist.astype(np.float32)
        hist /= (np.sum(hist) + 1e-8)
        hist_norm = hist.astype(np.float32)
        norm_val = np.linalg.norm(hist_norm)
        if norm_val > 0:
            hist_norm /= norm_val
        image_feats.append(hist_norm)
    # image_feats = np.matrix(image_feats)
    #print("image_feats.shape", image_feats.shape)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list,
        metric='cityblock'
    ):
    '''
    Args:
        train_img_feats (N, d): ndarray of feature of training images
        train_labels (N): list of string of ground truth category for each 
                          training image
        test_img_feats (M, d): ndarray of feature of testing images
    Returns:
        test_predicts (M): list of string of predict category for each 
                           testing image
    NOTE:
        1. d is the dimension of the feature representation, depending on using
           'tiny_image' or 'bag_of_sift'
        2. N is the total number of training images
        3. M is the total number of testing images
    '''

    ###########################################################################
    # TODO:                                                                   #
    # KNN predict the category for every testing image by finding the         #
    # training image with most similar (nearest) features, you can follow     #
    # below steps:                                                            #
    #   1. calculate the distance between training and testing features       #
    #   2. for each testing feature, select its k-nearest training features   #
    #   3. get these k training features' label id and vote for the final id  #
    # Remember to convert final id's type back to string, you can use CAT     #
    # and CAT2ID for conversion                                               #
    #                                                                         #
    # NOTE:                                                                   #
    # Some useful functions                                                   #
    #   Function : cdist(feats, feats)                                        #
    #                                                                         #
    # NOTE:                                                                   #
    #   1. instead of 1 nearest neighbor, you can vote based on k nearest     #
    #      neighbors which may increase the performance                       #
    #   2. hint: use 'minkowski' metric for cdist() and use a smaller 'p' may #
    #      work better, or you can also try different metrics for cdist()     #
    ###########################################################################

    test_predicts = []
    
    k = 4
    # dist = cdist(test_img_feats, train_feature)

    #     nearest = []
    #     distances = []
    label = np.array(train_labels) 
    # print('label arr:', label)
    dis = cdist(test_img_feats, train_img_feats, metric=metric)

    test_predicts = []
    for row in tqdm(dis, desc='inferencing...'):
        kmin = np.argpartition(row, k)[:k]
        # print("kmin:", kmin)
        labels = [CAT2ID[L] for L in label[kmin]]
        counts = np.bincount(labels)
        pred = np.argmax(counts)
        test_predicts.append(pred)

        # test_predicts.append( mode([ CAT2ID[L] for L in label[kmin] ]) )
        # print("label:", mode(label[kmin])[0][0])
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    # return [CAT2ID[cat] for cat in test_predicts]
    return [CAT[i] for i in test_predicts]
        ###########################################################################
    #                               END OF YOUR CODE                          #
    ###########################################################################
    