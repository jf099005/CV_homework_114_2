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
        ... (略過註解保持版面簡潔) ...
        """
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        guidance = guidance.astype(self.precision) / 255.0
        padded_guidance = padded_guidance.astype(self.precision) / 255.0

        if guidance.ndim == 2:
            guidance = guidance[:, :, np.newaxis]
            padded_guidance = padded_guidance[:, :, np.newaxis]

        # 確保 output 是正確的 float64 精度，避免後續型別轉換問題
        output = np.zeros(img.shape, dtype=self.precision)
        total_weights = np.zeros((img.shape[0], img.shape[1], 1), dtype=self.precision)
        
        H, W = img.shape[0], img.shape[1]
        C = guidance.shape[2]

        # --- 【核心優化】在迴圈外宣告緩衝區 (Buffers) ---
        buf_diff = np.empty((H, W, C), dtype=self.precision)
        buf_sum = np.empty((H, W, 1), dtype=self.precision)
        buf_img = np.empty_like(output)

        # 預先計算常數
        neg_two_sigma_r_sq = -2.0 * (self.sigma_r ** 2)

        for shift_y in range(-self.pad_w, self.pad_w + 1):
            for shift_x in range(-self.pad_w, self.pad_w + 1):
                Ly = self.pad_w + shift_y
                Ry = Ly + H
                Lx = self.pad_w + shift_x
                Rx = Lx + W
                
                # 取出對應的平移影像 (視圖 view，不產生新記憶體)
                shifted_img = padded_img[Ly:Ry, Lx:Rx]
                shifted_guidance = padded_guidance[Ly:Ry, Lx:Rx]

                # 1. 計算 guidance_diff 並寫入 buf_diff
                np.subtract(guidance, shifted_guidance, out=buf_diff)
                
                # 2. guidance_diff_sq 就地平方
                np.square(buf_diff, out=buf_diff)
                
                # 3. 沿通道加總，寫入 buf_sum (保留維度 keepdims=True 省去 np.newaxis)
                np.sum(buf_diff, axis=2, keepdims=True, out=buf_sum)

                # 4. 計算 range_kernel (就地運算)
                buf_sum /= neg_two_sigma_r_sq
                np.exp(buf_sum, out=buf_sum)

                # 5. 計算空間權重並乘上 range_kernel
                gaussian_weight = math.exp(-(shift_y**2 + shift_x**2) / (2 * (self.sigma_s**2)))
                buf_sum *= gaussian_weight  # 現在 buf_sum 等同於 combined_weight
                
                # 累加總權重
                total_weights += buf_sum

                # 6. 計算 weighted_img 並累加到 output
                np.multiply(buf_sum, shifted_img, out=buf_img)
                output += buf_img

        output /= total_weights
        return np.clip(output, 0, 255).astype(np.uint8)