import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        """
        Detect DoG keypoints from a grayscale image.

        Args:
            image (np.ndarray): Input grayscale image with shape (H, W) (np.float64).

        Returns:
            keypoints (np.ndarray): Array with shape (N, 2)
        """
        # TODO:
        # Step 1: Build Gaussian pyramid (2 octaves, 5 Gaussian images per octave)
        # - Use cv2.GaussianBlur(src, ksize=(0, 0), sigmaX=self.sigma**k).
        # - Octave 1 image shape: (H, W).
        # - Octave 2 base image: downsample octave 1's last image (sigma**4) by 2 using
        #   cv2.resize(..., interpolation=cv2.INTER_NEAREST),
        #   so shape becomes (H//2, W//2).

        # downsize_image = cv2.resize(image, image.shape[0]//2, image.shape[1]//2)

        gaussian_images = []


        iter_img = image

        for _ in range(self.num_octaves):
            # oct_gaussian_imgs_imgs = [iter_img]
            gaussian_images.append( np.ndarray( (self.num_guassian_images_per_octave, ) + iter_img.shape  ) )
            gaussian_filter_sigma = self.sigma
            for i_img in range(self.num_guassian_images_per_octave):
                # print(iter_img)
                iter_img = cv2.GaussianBlur(iter_img, ksize = (0,0), sigmaX = gaussian_filter_sigma)
                # print(iter_img)
                gaussian_images[-1][i_img] = iter_img
                # pass
                gaussian_filter_sigma = gaussian_filter_sigma*gaussian_filter_sigma

            iter_img = gaussian_images[-1][-1]
            iter_img = cv2.resize(iter_img, (iter_img.shape[1]//2, iter_img.shape[0]//2), interpolation = cv2.INTER_NEAREST)        
        print(len(gaussian_images[0]))
        print('img size:', [[img.shape for img in images] for images in gaussian_images])

        # Step 2: Build DoG pyramid (2 octaves, 4 DoG images per octave)
        # - For each octave, subtract adjacent Gaussian images:
        #   DoG_i = Gaussian_{i+1} - Gaussian_i.
        # - Use cv2.subtract(second_image, first_image).
        # - DoG image shape is the same as its corresponding Gaussian image.
        dog_images = []
        for octave_gaussian_imgs in gaussian_images:
            # oct_dog_imgs = []
            dog_images.append( np.ndarray((self.num_DoG_images_per_octave, ) + octave_gaussian_imgs[0].shape) )
            print(len(octave_gaussian_imgs))
            for i_img  in range(self.num_DoG_images_per_octave):
                # pass
                DoG_img = cv2.subtract( octave_gaussian_imgs[i_img + 1], octave_gaussian_imgs[i_img] )
                dog_images[-1][i_img] = DoG_img

            # dog_images.append( oct_dog_imgs )
        # Step 3: Threshold and find 3D local extrema in DoG volume
        # - Ignore 1-pixel image border.
        # - For each valid pixel in DoG images 1,2 of each octave, compare against
        #   its 26 neighbors in a 3x3x3 neighborhood.
        # - Keep [y, x] as a keypoint if:
        #   (1) it is a local maximum or minimum (>= max or <= min), and
        #   (2) abs(DoG value) > self.threshold.
        # - Coordinates stored in keypoints must be in original image scale:
        #   octave 1 -> [y, x], octave 2 -> [2*y, 2*x].
        keypoints = []
        print(len(dog_images))
        print(len(dog_images[0]))
        print([[img.shape for img in images] for images in dog_images])

        for i_octave in range(self.num_octaves):
            for i_img in range(1, len(dog_images[i_octave])-1):
                base_img = dog_images[i_octave][i_img]
                mask = base_img > self.threshold
                Y, X = np.where(mask)

                for y,x in zip(Y, X):
                    dog_value = base_img[y, x]
                    # for 
                    Ly, Ry = max([0, y-1]), min(base_img.shape[0], y+1)
                    Lx, Rx = max([0, x-1]), min(base_img.shape[1], x+1)
                    
                    try:
                        if dog_value >= np.max( dog_images[i_octave][i_img-1:i_img+1, Ly:Ry, Lx:Rx] ):
                        # continue
                            keypoints.append( (y,x) )
                    except Exception as e:
                        print(i_img, Ly, Ry, Lx, Rx)
                        print([(type(img), img.shape) for img in dog_images[i_octave][i_img-1:i_img+1]])
                        # print('cmp:', [img.shape for img in dog_images[i_octave][i_img-1:i_img+1][Ly:Ry, :]])
                        print('cmp:', dog_images[i_octave][i_img-1:i_img+1, Ly:Ry, Lx:Rx])
                        raise Exception(e)
                    


        # Step 4: Remove duplicate keypoints
        # - Use np.unique(..., axis=0).
        # - Expected shape after this step: (N, 2).
        keypoints = np.asarray(keypoints)
        keypoints = np.unique(keypoints, axis = 0)

        # Sort points using np.lexsort((col, row)) -> primary key col, secondary key row.
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints