#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
from loadpoints import load_velodyne_points

#camera calibration matrix as given 
cal_mat = np.array( [ [7.215377e+02, 0.000000e+00, 6.095593e+02], 
			  		[0.000000e+00, 7.215377e+02, 1.728540e+02],
			  		[0.000000e+00, 0.000000e+00, 1.000000e+00]  ] ) 

#Rotation matrix to go from the LiDAR frame to the camera frame
rot_mat = np.array( [ [0, -1, 0],
			  		[0, 0, -1],
			  		[1, 0, 0 ]  ] )
#translation matrix 
trans_mat = np.array( [ [-0.06, 0.08, 0.27] ])

Rt = np.hstack(( rot_mat, trans_mat.transpose() ))

points = load_velodyne_points('lidar-points.bin')
ones_mat = np.ones((6952,1))
points_mat = np.hstack(( points, ones_mat ))
# print points_mat.shape

temp = np.matmul(cal_mat, Rt)
final = np.matmul(temp, points_mat.transpose() )
final = final/final[2,:] #homogeneous coordinates

img = plt.imread("image.png")
plt.imshow(img)
plt.scatter(final[0,:], final[1,:], c=1/points[:,0], s=7);
plt.show()


