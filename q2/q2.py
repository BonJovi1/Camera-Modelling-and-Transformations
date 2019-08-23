#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import cv2

#camera calibration matrix as given 
cal_mat = np.array( [ [7.2153e+02, 0, 6.0955e+02], 
			  		[0, 7.2153e+02, 1.7285e+02],
			  		[0, 0, 1]  ] ) 
cam_inverse = np.linalg.inv(cal_mat)

#found using imtool in matlab 
pixels = np.array( [ [825, 308, 1] ] ) 

temp = np.matmul(cam_inverse, pixels.transpose())
# print(temp); #temp is coordinates in camera plane, but no depth, z = fy. Z = Y.fy / y

fy = 7.2153e+02
Y = 312
y = 1.65

z = (fy*y)/(Y - 1.7285e+02)
# z = (y * fy)/Y
# print(z);

temp = z * temp
# print(temp);#camera cooridnates of origin

X = temp[0]
Y = temp[1]
Z = temp[2]

height = 1.38
width = 1.51
length = 4.10
flb = np.array([X, Y, Z])
frb = np.array([X+ width, Y, Z])
flt = np.array([X, Y-height, Z])
frt = np.array([X+width, Y-height, Z])

nlb = np.array([X, Y, Z+length])
nrb = np.array([X+width, Y, Z+length])
nlt = np.array([X, Y-height, Z+length])
nrt = np.array([X+width, Y-height, Z+length])

flb2 = np.matmul(cal_mat, flb)
frb2 = np.matmul(cal_mat, frb)
flt2 = np.matmul(cal_mat, flt)
frt2 = np.matmul(cal_mat, frt)
nlb2 = np.matmul(cal_mat, nlb)
nrb2 = np.matmul(cal_mat, nrb)
nlt2 = np.matmul(cal_mat, nlt)
nrt2 = np.matmul(cal_mat, nrt)

flb2 = flb2/flb2[2]
frb2 = frb2/frb2[2]
flt2 = flt2/flt2[2]
frt2 = frt2/frt2[2]
nlb2 = nlb2/nlb2[2]
nrb2 = nrb2/nrb2[2]
nlt2 = nlt2/nlt2[2]
nrt2 = nrt2/nrt2[2]

print(flb2)
print(frb2)
print(flt2)
print(frt2)


connect = [(flb2, frb2, flt2, nlb2), (frt2, flt2, frb2, nrt2), (nlt2, flt2, nlb2, nrt2), (nrb2, nrt2, nlb2, frb2)]

def main():
    image = plt.imread("image.png")

    for group in connect:
        pt1 = group[0]
        for i in range(1,4):
            pt =  group[i]
             
            plt.plot([pt1[0], pt[0]], [pt1[1], pt[1]], c='r', linewidth = 5)

    plt.imshow(image)
    plt.show()

main()


