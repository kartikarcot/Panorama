import numpy as np
import cv2
import matplotlib.pyplot as plt

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = []
    
    ##################
    # TO DO ...
    # Compute principal curvature here
    h,w,levels = np.shape(DoG_pyramid)
    for i in range(0,levels):
        im = DoG_pyramid[:,:,i]
        
        
        D_xx = cv2.Sobel(im,ddepth = -1,dx = 2,dy = 0)
        D_yy = cv2.Sobel(im,ddepth = -1,dx = 0,dy = 2)
        D_xy = cv2.Sobel(im,ddepth = -1,dx = 1,dy = 1)
        Det = D_xx*D_yy - D_xy*D_xy
        
#         D_x = cv2.Sobel(im,ddepth = -1,dx = 1,dy = 0)
#         D_y = cv2.Sobel(im,ddepth = -1,dx = 0,dy = 1)
    
#         D_xx = cv2.Sobel(D_x,ddepth = -1,dx = 1,dy = 0)
#         D_yy = cv2.Sobel(D_y,ddepth = -1,dx = 0,dy = 1)
        
#         D_xy = cv2.Sobel(D_y,ddepth = -1,dx = 1,dy = 0)
#         D_yx = cv2.Sobel(D_x,ddepth = -1,dx = 0,dy = 1)
        
#         Det = D_xx*D_yy - D_xy*D_yx
        
        Trace = D_xx + D_yy
        R = Trace**2/(Det+1e-05)
        principal_curvature.append(R.copy())
    
    return np.moveaxis(np.array(principal_curvature), 0,-1)


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = []
    
    ##############
    #  TO DO ...
    # Compute locsDoG here
    contrast_mask = abs(DoG_pyramid) > th_contrast
    principal_mask = abs(principal_curvature) < th_r
    threshold_mask = contrast_mask*principal_mask

    h_im,w_im,l = np.shape(DoG_pyramid)
    pruned_mask = np.zeros((h_im,w_im,l), dtype=bool)
    h_arr, w_arr, z_arr = np.nonzero(threshold_mask)
    # print(len(list(x_arr)))
    for i in range(0,len(list(h_arr))):
        h = h_arr[i]
        w = w_arr[i]
        z = z_arr[i]
        up = max(0,z-1)
        down = min(l-1,z+1)
        left = max(0,w-1)
        right = min(w_im-1,w+1)
        top = max(0,h-1)
        bottom = min(h_im-1,h+1)
#         locsDoG.append(np.array([x,y,z]))
#         
        a = max(np.max(DoG_pyramid[top:bottom+1,left:right+1,z]), np.max(DoG_pyramid[h,w,up:down+1]))
#     
        b = min(np.min(DoG_pyramid[top:bottom+1,left:right+1,z]), np.min(DoG_pyramid[h,w,up:down+1]))
        
        if DoG_pyramid[h,w,z] == a:
            pruned_mask[h,w,z] = True
            locsDoG.append(np.array([w,h,z]))

        if DoG_pyramid[h,w,z] == b:
            pruned_mask[h,w,z] = True
            locsDoG.append(np.array([w,h,z]))

    # print(np.sum(pruned_mask.astype(np.int8)))
    # print(np.shape(np.array(locsDoG)))
    return np.array(locsDoG)
def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, locsDoG here
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)

    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)

    # # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)

    # # test get local extrema
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    return locsDoG, im_pyr

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    '''
    
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    
    DoG_levels = levels[1:]
    for i in range(1,len(levels)):
        DoG_pyramid.append(gaussian_pyramid[:,:,i-1]-gaussian_pyramid[:,:,i])
    DoG_pyramid = np.moveaxis(np.array(DoG_pyramid), 0,-1)
    # print(np.shape(DoG_pyramid))
    return DoG_pyramid, DoG_levels


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
#    im = cv2.imread('../data/model_chickenbroth.jpg')
    im = cv2.imread('../data/chickenbroth_01.jpg')
    im_pyr = createGaussianPyramid(im)
#    displayPyramid(im_pyr)
#    plt.imsave("im_pyr.jpg",im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
#    displayPyramid(DoG_pyr)
#    plt.imsave("dog.jpg",DOG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    locsDoG, gaussian_pyramid = DoGdetector(im,th_contrast = 0.03,th_r = 12)
    N,_ = np.shape(locsDoG)
    for i in range(0,N):
        im = cv2.circle(im,(locsDoG[i][0],locsDoG[i][1]),radius=1, color=(0,255,0),thickness=-1)
#    plt.imsave("interest_points.jpg",im)
    cv2.imwrite('interest_points.jpg',im)
