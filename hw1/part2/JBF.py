import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        """
        Parameters
        ----------
        img : ndarray
            The image to be filtered.

        guidance : ndarray
            The guidance image used to compute range weights.
            It can be either:
              - grayscale: shape (H, W)
              - color:     shape (H, W, C)

        Returns
        -------
        output : ndarray
            The filtered result with the same shape as img.
        """
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # Note: Pixel values should be normalized to [0, 1] (divided by 255) to construct range kernel.
            
        return np.clip(output, 0, 255).astype(np.uint8)