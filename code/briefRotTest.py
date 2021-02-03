import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from BRIEF import briefLite,briefMatch,plotMatches


# test briefLite
im = cv2.imread('../data/model_chickenbroth.jpg')
# im = cv2.imread('../data/chickenbroth_01.jpg')
match_arr = []
mismatch_arr = []

for theta in range(-90,90,10):
    # print("theta is" + str(theta))
    theta = theta
    rows,cols,_ = np.shape(im)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
    dst = cv2.warpAffine(im,M,(cols,rows))
    locs1, desc1 = briefLite(im)
    locs2, desc2 = briefLite(dst)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im,dst,matches,locs1,locs2)
    # locs1[:,0],locs1[:,1] = locs1[:,1], locs1[:,0]
    # locs2[:,0],locs2[:,1] = locs2[:,1], locs2[:,0]
    # plotMatches(im,dst,matches,locs1,locs2)
    # print(matches.shape)
    M_inv = cv2.getRotationMatrix2D((cols/2,rows/2),-theta,1)
    matched=0
    mismatched=0
    for i in range(0, matches.shape[0]):
        dst_point = locs2[matches[i][1]]
        src_point = locs1[matches[i][0]]
        dst_point[2] = 1
        dst_mapped = np.floor(M_inv.dot(dst_point.T)).T
        if np.sum((dst_mapped-src_point[:2])*(dst_mapped-src_point[:2])) < 2:
            matched+=1
        else:
            mismatched+=1
    match_arr.append(matched)
    mismatch_arr.append(mismatched)

angles = [i for i in range(-90,90,10)]
total = [mismatch_arr[i]+match_arr[i] for i in range(len(match_arr))]
plt.clf()
plt.bar(angles,total,width=3.2)
plt.title("Total Number of Matches vs Rotation Angle")
plt.xlabel('Angle')
plt.ylabel('No of Matches')
plt.savefig('matches.png')
plt.clf()
plt.bar(angles,match_arr,width=3.2)
plt.title("Total Number of Correct Matches vs Rotation Angle")
plt.xlabel('Angle')
plt.ylabel('No of Correct Matches')
plt.savefig('correct_matches.png')