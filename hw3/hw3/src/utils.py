import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    for i, (ui, vi) in enumerate(zip(u, v)):
        v1,v2 = vi
        v3 = 1
        ui = np.append(ui, 1)
        Ai = np.array(
            [
                np.concatenate([np.zeros(3), -v3*ui, v2*ui]),
                np.concatenate([v3*ui, np.zeros(3), -v1*ui]),
                np.concatenate([-v2*ui, v1*ui, np.zeros(3)])
            ]
        )

        A[2*i:2*i+2, :] = Ai[:2, :]


    # TODO: 2.solve H with A
    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1, :].reshape(3, 3)/Vh[-1, -1]
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    
    # TODO: 1.meshgrid the (x,y) coordinate pairs
    src_x, src_y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    src_x = src_x.flatten()
    src_y = src_y.flatten()
    src_M = np.stack((src_x, src_y, np.ones_like(src_x.flatten())), axis=1)
    src_M = src_M.T

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # H = np.linalg.inv(H)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking

        M_dst = np.matmul(H, src_M)
        M_dst = M_dst/M_dst[2, :]
        print("M_dst shape: ", M_dst.shape)
        grid_dst = M_dst.T.astype(int)#.reshape(h_dst, w_dst, 1)
        print('shape of grid_dst:', grid_dst.shape)
        print("first 5:", grid_dst[:5])
        print('src shape:', src_x.shape, src_y.shape)
        for (x,y, _), sx, sy in zip(grid_dst, src_x, src_y):
            # if xmin <= x < xmax and ymin <= y < ymax:
                # print(y, x, sy, sx)
                src[sy, sx] = dst[y, x]
        return src


    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing

        print(H.shape, src_M.shape)
        M_dst = np.matmul(H, src_M)
        M_dst = M_dst/M_dst[2, :]
        print("M_dst shape: ", M_dst.shape)
        grid_dst = M_dst.T.astype(int)#.reshape(h_dst, w_dst, 1)
        print('shape of grid_dst:', grid_dst.shape)
        print("first 5:", grid_dst[:5])
        print('src shape:', src_x.shape, src_y.shape)
        for (x,y, _), sx, sy in zip(grid_dst, src_x, src_y):
            # if xmin <= x < xmax and ymin <= y < ymax:
                # print(sy, sx)
            dst[y, x] = src[sy, sx]
        return dst
