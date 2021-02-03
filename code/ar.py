import numpy as np
import cv2
import os
from planarH import computeH


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    
    H_inv = np.linalg.inv(K)@H
    # print(H_inv)
    U, S, Vh = np.linalg.svd(H_inv[:,:2])
    diag = np.array([[1, 0], [0,1], [0,0]])
    rotation = U@diag@Vh
    # print(rotation)
    r3 = np.cross(rotation[:,0],rotation[:,1], axis=0)
    # print(r3)
    # r3 = r3/np.sum(r3**2)
    
#    print(np.sum(r3**2))
#     print(r3.shape)
#     print(len(r3))
    r3 = np.reshape(r3, (len(r3),1))
    rot_matrix = np.hstack((rotation, r3))
    
#     print(rot_matrix.shape)
#    print(rot_matrix)
#    print(np.linalg.det(rot_matrix))
#    print(np.sum(rot_matrix**2))
    rot_matrix[:,2] = -1*rot_matrix[:,2] if np.linalg.det(rot_matrix) < 0 else rot_matrix[:,2]
    
    arb_scale = np.mean(H_inv[:,:2]/rot_matrix[:,:2])
    # print(arb_scale)
#     print(rot_matrix)
    R, t = rot_matrix, (H_inv/arb_scale)[:,2]
    # print(R)
    return R, t

def project_extrinsics(K, W, R, t):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        W - 3D planar points of textbook
        R - relative 3D rotation
        t - relative 3D translation
    OUTPUTS:
        X - computed projected points
    '''

    #############################
    # TO DO ...
    H_new = np.hstack((R,np.reshape(t,(len(t),1))))
#    print(H_new)
#    print(W)
#    print(W.shape)
    homo_points = K@H_new@W
    homo_points = homo_points/(homo_points[2,:])
    return homo_points


if __name__ == "__main__":
    im = cv2.imread('../data/prince_book.jpeg')
    f = open('../data/sphere.txt', 'r')
    a = f.read().split('\n')
    x = a[0].split('  ')[1:]
    y = a[1].split('  ')[1:]
    z = a[2].split('  ')[1:]
    x_vals = [float(i) for i in x]
    y_vals = [float(i) for i in y]
    z_vals = [float(i) for i in z]

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)
    

    X = np.array([[483, 1704, 2175, 67], [810, 781, 2217, 2286]])
    K = np.array([[3043.72, 0.0, 1196.0], [0.0, 3043.72, 1604.00], [0.0, 0.0, 1.0]])
    W = np.array([[0.0, 18.2, 18.2, 0.0], [0.0, 0.0, 26.0, 26.0], [0.0, 0.0, 0.0, 0.0]])
    H = computeH(X,W)

    H_inv = computeH(W[:2],X)
    o_xy = np.reshape(np.array([830,1640,1]),(3,1))
    o_world = H_inv@o_xy
    o_world = o_world/o_world[2]
#    print(o_world)
    z_vals = z_vals+np.min(z_vals)
    x_vals = x_vals+o_world[0]
    y_vals = y_vals+o_world[1]
    points = np.vstack((x_vals,y_vals,z_vals, np.ones((1,len(x_vals)))))

#    print('Homography')
#    print(H)
    R,t = compute_extrinsics(K,H)
#    print('Translation')
#    print(t)
#    print('Rotation')
#    print(R)
    homo_points = project_extrinsics(K,points,R,t)
    homo_points = homo_points.T
    N,_ = np.shape(homo_points)
#    print(homo_points)
    for i in range(0,N):
        im = cv2.circle(im,(int(homo_points[i][0]),int(homo_points[i][1])),radius=5, color=(0,255,0),thickness=-1)

    cv2.imwrite('ar1.jpg', im)
        # plt.imshow(im)
        #############################
        # TO DO ...
        # perform required operations and plot sphere
        
