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
        output = img.astype(self.precision)
        total_weights = np.ones((img.shape[0], img.shape[1], 1), dtype=self.precision)
        img_height = img.shape[0]
        img_width = img.shape[1]
        print('shape', img.shape)
        print('padding', self.pad_w)
        range_kernel_pool = np.square(\
                    padded_guidance[:padded_guidance.shape[0] - shift_y, Ln:padded_guidance.shape[1]] - \
                    padded_guidance[    :, Lp:padded_guidance.shape[1]]
                )

        for shift_y in range(self.pad_w+1):
            for shift_x in range(-self.pad_w, self.pad_w+1):
                if shift_y == 0 and shift_x <= 0:
                    continue
                gaussian_weight = math.exp(-(shift_y**2 + shift_x**2)/(2*(self.sigma_s**2)))

                Lp = max([0, shift_x])
                Ln = max([0, -shift_x])
                
                # guidance_diff =\
                #     padded_guidance[:padded_guidance.shape[0] - shift_y, Ln:padded_guidance.shape[1] - Lp]\
                #     - padded_guidance[shift_y:, Lp:padded_guidance.shape[1] - Ln]
                
                # guidance_diff_sq = guidance_diff*guidance_diff

                # guidance_diff_sq = np.sum(guidance_diff_sq, axis=2)
                # range_kernel_pool = np.exp(-guidance_diff_sq/(2*(self.sigma_r**2)))[:,:,np.newaxis]


                range_kernel_pool = np.square(\
                    padded_guidance[:padded_guidance.shape[0] - shift_y, Ln:padded_guidance.shape[1] - Lp] - \
                    padded_guidance[shift_y:, Lp:padded_guidance.shape[1] - Ln]
                )

                range_kernel_pool = np.sum(range_kernel_pool, axis=2, keepdims=True)
                range_kernel_pool = gaussian_weight*np.exp(-range_kernel_pool/(2*(self.sigma_r**2)))#[:,:,np.newaxis]
                                                
                range_kernel1 = range_kernel_pool[
                    self.pad_w - shift_y: self.pad_w - shift_y + img_height ,\
                    self.pad_w - Lp: self.pad_w - Lp + img_width
                ]
                # range_kernel1 = range_kernel1[:,:,np.newaxis]

                range_kernel2 = range_kernel_pool[
                    self.pad_w: self.pad_w + img_height,
                    self.pad_w - Ln: self.pad_w -Ln + img_width
                ]  
                # range_kernel2 = range_kernel2[:,:,np.newaxis]

                L1 = self.pad_w - shift_x
                L2 = self.pad_w + shift_x

                output += range_kernel1*padded_img[self.pad_w - shift_y: self.pad_w - shift_y + img_height, L1: L1 + img_width] + \
                               range_kernel2*padded_img[self.pad_w + shift_y: self.pad_w + shift_y + img_height, L2: L2 + img_width]
                # output += weighted_img
                total_weights += range_kernel1 + range_kernel2
                
                # weighted_img *= gaussian_weight
                # output += weighted_img
                # total_weights += gaussian_weight*(range_kernel1 + range_kernel2)

        output /= total_weights
        result = np.clip(output, 0, 255).astype(np.uint8)
        
        # Fix 3: Squeeze the output back to 2D if the input was grayscale
        # if is_grayscale_img:
        #     result = np.squeeze(result, axis=2)
            
        return result