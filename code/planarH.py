import numpy as np
import cv2
from BRIEF import briefLite, briefMatch


def computeH(p1, p2):
    '''
    INPUTS
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                coordinates between two images
    OUTPUTS
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
                equation
    '''

    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    H2to1 = None
    # TO DO ...
    A = []
    for i in range(0, p1.shape[1]):
        u, v = p2[0,i],p2[1,i]
        x, y = p1[0,i],p1[1,i]
        a = [0,0,0, -u, -v, -1, y*u, y*v, y]
        b = [u, v, 1, 0, 0, 0, -x*u, -x*v, -x]
        A.append(a)
        A.append(b)
    A = np.array(A)
    # print(A)
    # print(A.shape)
    u , s, vh = np.linalg.svd(A)
    # print(vh.shape)
    # print(u.shape)
    # print(s.shape)
    H2to1 = np.reshape(vh.T[:,-1], (3,3))
    H2to1 = H2to1/H2to1[-1,-1]
    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using RANSAC
    
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches         - matrix specifying matches between these two sets of point locations
        nIter           - number of iterations to run RANSAC
        tol             - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    bestH = None
    ###########################
    # TO DO ...
    # print(matches.shape)
    X1 = locs1[matches[:,0],:].T
    X2 = locs2[matches[:,1],:].T

    X1 = X1[:2]
    X2 = X2[:2]
    X2 = X2[:2]
    H = computeH(X1,X2)

    X1_homog = np.vstack((X1, np.ones((1, X1.shape[1]))))
    X2_homog = np.vstack((X2, np.ones((1, X2.shape[1]))))
    # print(X1.shape)
    # print(X2.shape)

    
    # X1_pred = X1_pred/X1_pred[2,:]
    # print(X1_pred[:,0])
    mismatches_min = X1.shape[1]+2
    # print("min is " + str(mismatches_min))
    # sleep(10)
    H_min = None
    for k in range(0, num_iter):
        choices = np.random.randint(0,X1.shape[1], size = (4,))
        X1_selected = X1[:,choices]
        X2_selected = X2[:,choices]
        # print(X2_selected)
        # print(X1_homog.shape)
        H = computeH(X1_selected, X2_selected)
        # print("Homography")
        # print(H)
        X1_pred = np.matmul(H,X2_homog)
        X1_pred = X1_pred/((X1_pred[2,:])+1e-5)
        error = np.sum((X1 - X1_pred[:2,:])**2, axis = 0)

        error_bool = error > tol**2
        # print(error_bool.shape)
        mismatches = np.sum(error_bool.astype(np.int8))

        if mismatches < mismatches_min:
            mismatches_min = mismatches
            # print("min is " + str(mismatches_min))
            H_min = H
    return H_min


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    # t=time.time()
    matches = briefMatch(desc1, desc2)
    # print(time.time()-t)
    print(ransacH(matches, locs1, locs2, num_iter=5000, tol=2))
