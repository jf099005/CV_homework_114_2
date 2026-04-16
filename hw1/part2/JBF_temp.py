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
        output = np.zeros((img.shape))
        total_weights = np.zeros((img.shape[0], img.shape[1], 1), dtype=self.precision)

            # gaussian_weights = [[math.exp(-(a**2 + b**2)/(2*(self.sigma_s**2))) for b in range(3+1)] for a in range(3+1)]

        # y, x = np.ogrid[-self.pad_w : self.pad_w + 1, -self.pad_w : self.pad_w + 1]
        # spatial_kernel = np.exp(-(x**2 + y**2) / (2 * (self.sigma_s**2)))
        # y = np.ogrid
        # gaussian_1d_weights = np.exp( -x**2/(2*self.sigma_s**2) )

        for shift_y in range(-self.pad_w, self.pad_w+1):
            # Y_shift_weights = np.zeros((img.shape[0], img.shape[1], 1), dtype=self.precision)
            
            for shift_x in range(-self.pad_w, self.pad_w+1):
                Ly = self.pad_w + shift_y
                Ry = Ly + img.shape[0]
                Lx = self.pad_w + shift_x
                Rx = Lx + img.shape[1]
                shifted_img = padded_img[Ly:Ry, Lx:Rx]

                guidance_diff = guidance - padded_guidance[Ly:Ry, Lx:Rx]
                guidance_diff_sq = guidance_diff*guidance_diff
                guidance_diff_sq = np.sum(guidance_diff_sq, axis=2)
                # gaussian_weight = gaussian_weights[abs(shift_x)][abs(shift_y)]#
                # math.exp(-(shift_y**2 + shift_x**2)/(2*(self.sigma_s**2)))
                gaussian_weight = math.exp(-(shift_y**2 + shift_x**2)/(2*(self.sigma_s**2)))
                
                range_kernel = np.exp(-guidance_diff_sq/(2*(self.sigma_r**2)))
                range_kernel = range_kernel[:,:,np.newaxis]
                weighted_img = gaussian_weight*range_kernel*shifted_img
                output += weighted_img
                total_weights += gaussian_weight*range_kernel

        output /= total_weights
        return np.clip(output, 0, 255).astype(np.uint8)