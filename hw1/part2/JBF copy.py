import numpy as np
import cv2
import math

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        self.precision = np.float64
    
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

        guidance = guidance.astype(self.precision)/255.0
        padded_guidance = padded_guidance.astype(self.precision)/255.0

        if guidance.ndim == 2:
            guidance = guidance[:,:,np.newaxis]
            padded_guidance = padded_guidance[:,:,np.newaxis]

            # guidance = np.expand_dims(guidance, 2)
        ### TODO ###
        # Note: Pixel values should be normalized to [0, 1] (divided by 255) to construct range kernel.
        # output = np.zeros((img.shape))
        output = img.astype(self.precision)
        # total_weights = np.zeros((img.shape[0], img.shape[1], 1), dtype=self.precision)
        total_weights = np.ones((img.shape[0], img.shape[1], 1), dtype=self.precision)

            # gaussian_weights = [[math.exp(-(a**2 + b**2)/(2*(self.sigma_s**2))) for b in range(3+1)] for a in range(3+1)]

        # y, x = np.ogrid[-self.pad_w : self.pad_w + 1, -self.pad_w : self.pad_w + 1]
        # spatial_kernel = np.exp(-(x**2 + y**2) / (2 * (self.sigma_s**2)))
        # y = np.ogrid
        for shift_y in range(self.pad_w+1):
            
            for shift_x in range(self.pad_w+1):
                if(shift_y == 0 and shift_x == 0):
                    continue
                # Ly = self.pad_w + shift_y
                # Ry = Ly + img.shape[0]
                # Lx = self.pad_w + shift_x
                # Rx = Lx + img.shape[1]
                # shifted_img = padded_img[Ly:Ry, Lx:Rx]
                # print('shape of padded guidance:', padded_guidance.shape)
                # print('shift:', shift_y, shift_x)
                guidance_diff =\
                    padded_guidance[:padded_guidance.shape[0] - shift_y, :padded_guidance.shape[0] - shift_x]\
                    - padded_guidance[shift_y:, shift_x:]
                # print('shape of gd:', guidance_diff.shape)
                # print('shape of img:', img.shape)
                # guidance_diff = guidance - padded_guidance[Ly:Ry, Lx:Rx]
                guidance_diff_sq = guidance_diff*guidance_diff
                guidance_diff_sq = np.sum(guidance_diff_sq, axis=2)
                # gaussian_weight = gaussian_weights[abs(shift_x)][abs(shift_y)]#
                # math.exp(-(shift_y**2 + shift_x**2)/(2*(self.sigma_s**2)))
                gaussian_weight = math.exp(-(shift_y**2 + shift_x**2)/(2*(self.sigma_s**2)))
                range_kernel_pool = np.exp(-guidance_diff_sq/(2*(self.sigma_r**2)))
                range_kernel1 = range_kernel_pool[
                    self.pad_w - shift_y: self.pad_w - shift_y + img.shape[0] ,\
                    self.pad_w - shift_x: self.pad_w - shift_x + img.shape[1]
                ]#np.exp(-guidance_diff_sq/(2*(self.sigma_r**2)))
                range_kernel1 = range_kernel1[:,:,np.newaxis]

                range_kernel2 = range_kernel_pool[
                    self.pad_w: self.pad_w + img.shape[0],
                    self.pad_w: self.pad_w + img.shape[1]
                ]  #np.exp(-guidance_diff_sq/(2*(self.sigma_r**2)))
                range_kernel2 = range_kernel2[:,:,np.newaxis]

                weighted_img = range_kernel1*padded_img[shift_y: shift_y + img.shape[0], shift_x: shift_x + img.shape[1]] + \
                               range_kernel2*padded_img[self.pad_w + shift_y: self.pad_w + shift_y + img.shape[0], self.pad_w + shift_x: self.pad_w + shift_x + img.shape[1]]
                weighted_img *= gaussian_weight
                output += weighted_img
                total_weights += gaussian_weight*(range_kernel1 + range_kernel2)

        output /= total_weights
        return np.clip(output, 0, 255).astype(np.uint8)