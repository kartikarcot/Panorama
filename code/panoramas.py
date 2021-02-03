import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt
import time

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################
    # TO DO ...
    
    pano_im = cv2.warpPerspective(im2, H, dsize = (im2.shape[1]+im1.shape[1],im2.shape[0]))
    h,w,_ = im1.shape
    # cv2.imwrite('R_sticth.jpg',pano_im)
    # print(np.maximum(im1,pano_im[:h,:w]))
    pano_im[:h,:w] = np.maximum(im1,pano_im[:h,:w])
    # cv2.imwrite('pano_sticth.jpg',pano_im)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    pano_im = None
    ######################################
    # TO DO ...

    a2_x = np.array([0, im2.shape[1], 0, im2.shape[1]])
    a2_y = np.array([0,0,im2.shape[0], im2.shape[0]])
    a2_1 = np.array([1,1,1,1])
    a2 = np.vstack([a2_x, a2_y,a2_1])
    x2 = np.matmul(H2to1, a2)
    x2_norm = x2/x2[2]

    x2_max_x_norm = np.max(x2_norm[0])
    x2_max_y_norm = np.max(x2_norm[1])
    
    x2_min_x_norm = np.min(x2_norm[0])
    x2_min_y_norm = np.min(x2_norm[1])
    
    x1_max = np.array([[im1.shape[1]],
                        [im1.shape[0]],
                        [1]])
    
    x1_min = np.array([[0],
                        [0],
                        [1]])
    
    
    '''
    x1_max = np.matmul(H2to1, x1_max)
    x1_max_norm = x1_max/x1_max[2]

    x1_min = np.matmul(H2to1, x1_min)
    x1_min_norm = x1_min/x1_min[2]
    '''
    out_x_max = max(x1_max[0], x2_max_x_norm)
    out_x_min = min(x1_min[0], x2_min_x_norm)

    out_y_max = max(x1_max[1], x2_max_y_norm)
    out_y_min = min(x1_min[1], x2_min_y_norm)
    # print([out_x_max, out_y_max])
    # print([out_x_min, out_y_min])
    out_x = out_x_max - out_x_min
    out_y = out_y_max - out_y_min
    error = 0
    out_x += error
    out_y += error
    # print(out_x)
    # print(out_y)
    tx = abs(out_x_min) + error
    ty = abs(out_y_min) + error

    #arbitrary scaling to ensure no clipping
    alpha = 1
    M = np.array([[alpha, 0, tx],
                [0,alpha,ty],
                [0,0,1]], dtype=float)
    im2_transformed = cv2.warpPerspective(im2, np.matmul(M,H2to1), dsize = (int(out_x), int(out_y)))
    im1_transformed = cv2.warpPerspective(im1, M, dsize = (int(out_x), int(out_y)))
    # cv2.imwrite('R.jpg',im2_transformed)
    # cv2.imwrite('L.jpg',im1_transformed)
    pano_im = np.maximum(im1_transformed, im2_transformed)
    return pano_im


def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    t = time.time()
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    print('Time to compute Brieflite: '+str(time.time()-t))
    t = time.time()
    matches = briefMatch(desc1, desc2)
    print('Time to compute BriefMatch: '+ str(time.time()-t))
    t = time.time()
    H = ransacH(matches, locs1, locs2, num_iter=2000, tol=2)
    print('Time to compute Ransac: '+str(time.time()-t))
    np.save('../results/q6_1.npy',H)
    return imageStitching_noClip(im1,im2,H)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
#    locs1, desc1 = briefLite(im1)
#    locs2, desc2 = briefLite(im2)
#    matches = briefMatch(desc1, desc2)
#    H = ransacH(matches, locs1, locs2, num_iter=2000, tol=2)
#    # print(H)
#    np.save('q6_1.npy',H)
    # imageStitching(im1,im2,H)
    t = time.time()
    cv2.imwrite('pano.jpg',generatePanaroma(im1, im2))
    print('Time to compute panorama: ' + str(time.time()-t))
    # cv2.imwrite('pano.jpg',imageStitching(im1,im2,H))

    # cv2.imwrite('pano1.jpg',imageStitching_noClip(im1,im2,H))

