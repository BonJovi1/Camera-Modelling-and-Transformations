# Assignment 1, Question 3

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Info Given

## Tag size = 0.1315 m

## corner pixel locations (clock-wise from top-left to bottom-right):

# tag-id 24 (left)
# u v
# 284.56243896 149.2925415
# 373.93179321 128.26719666
# 387.53588867 220.2270813
# 281.29962158 241.72782898

# #tag-id 25 (right)
# u v
# 428.86453247 114.50731659
# 524.76373291 92.09218597
# 568.3659668 180.55757141
# 453.60995483 205.22370911

## Distance between corner 2 of tag-1 and corner 1 of tag-2: 0.0790 m


# Utility Functions

def display_image(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.show()

def draw_line(img, A, B):
    cv2.line(img, A, B, (0, 255, 0), thickness=3, lineType=8)

def add_column(original, newcol):
    return np.concatenate([original, newcol], axis=1)

def add_row(original, newrow): # didn't use
    return np.concatenate(original, newrow, axis=0)

def camera_project(P, world_point):
    # Convert the points in 3d space to homogeneous coordinates by appending 1, and making it a column vector
    homo_point = np.transpose(np.array([world_point[0], world_point[1], world_point[2], 1]))

    #Projecting the points into 2d by applying the camera projection matrix
    val = np.matmul(P, homo_point)

    # Returned pixel coordinates, homogenous(a, b, c) => (a/c, b/c) in pixels.
    # int as pixels are integral.
    return (int(val[0] / val[2]), int(val[1] / val[2]))

def compute_homography(lstpts): # Calculates homography matrix b/w initial and final points
    x1 = lstpts[0][0][0]
    x2 = lstpts[1][0][0]
    x3 = lstpts[2][0][0]
    x4 = lstpts[3][0][0]
    y1 = lstpts[0][0][1]
    y2 = lstpts[1][0][1]
    y3 = lstpts[2][0][1]
    y4 = lstpts[3][0][1]
    x1f = lstpts[0][1][0]
    x2f = lstpts[1][1][0]
    x3f = lstpts[2][1][0]
    x4f = lstpts[3][1][0]
    y1f = lstpts[0][1][1]
    y2f = lstpts[1][1][1]
    y3f = lstpts[2][1][1]
    y4f = lstpts[3][1][1]
    ans = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    q = np.array([
        [-1*x1, -y1, -1, 0, 0, 0, x1*x1f, y1*x1f, x1f],
        [0, 0, 0, -1*x1, -1*y1, -1, x1*y1f, y1*y1f, y1f],
        [-1*x2, -y2, -1, 0, 0, 0, x2*x2f, y2*x2f, x2f],
        [0, 0, 0, -1*x2, -1*y2, -1, x2*y2f, y2*y2f, y2f],
        [-1*x3, -y3, -1, 0, 0, 0, x3*x3f, y3*x3f, x3f],
        [0, 0, 0, -1*x3, -1*y3, -1, x3*y3f, y3*y3f, y3f],
        [-1*x4, -y4, -1, 0, 0, 0, x4*x4f, y4*x4f, x4f],
        [0, 0, 0, -1*x4, -1*y4, -1, x4*y4f, y4*y4f, y4f],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    H = list(np.linalg.solve(q,ans))
    H = np.array([H[:3], H[3:6], H[6:9]])
    return H

def draw_point(image, point, color):
    if color == "green":
        cv2.circle(image, point, 10, (0, 255, 0), thickness=3, lineType=8)
    elif color == "red":
        cv2.circle(image, point, 8, (0, 0, 255), thickness=3, lineType=8)


#
# To frame the question, normally we have carmera matrix P * World_Point to give image coordinate.
# Issue here is that it won't as a Homography H has been applied on the image. Our task is to recover the homography matrix, and thus figuring out the pose of the camera.
#
# We want:
# Img_coord = H * P * World_Point
#
# Here we know the Img_coords and P. World Points can be estimated given that we know the size of the tags.
#

# Uncomment for printing original image
# image = cv2.imread("./image.png")
# display_image(image)


# Convention: l/r.  t/b.   l/r    => (l)eft/(r)ight tag,   (t)op/(b)ottom side,    (l)eft/(r)ight corner
# c in front => (c)orresponding pixel location
X = 0
Y = 0
Z = 0
# Origin chosen as top left of left april tag
size = 0.1315 # Value Given
ltl = (X, Y, Z)
cltl = (284.56243896, 149.2925415)
ltr = (X+size, Y, Z)
cltr = (373.93179321, 128.26719666)
lbl = (X, Y-size, Z)
clbl = (281.29962158, 241.72782898)
lbr = (X+size, Y-size, Z)
clbr = (387.53588867, 220.2270813)

X2 = X + size + 0.0790 # Value given
rtl = (X2, Y, Z)
crtl = (428.86453247, 114.50731659)

rtr = (X2+size, Y, Z)
crtr = (524.76373291, 92.09218597)
rbl = (X2, Y-size, Z)
crbl = (453.60995483, 205.22370911)
rbr = (X2+size, Y-size, Z)
crbr = (568.3659668, 180.55757141)
world_pts = [ltl, ltr, lbl, lbr, rtl, rtr, rbl, rbr]
original_pixel_pts = [cltl, cltr, clbl, clbr, crtl, crtr, crbl, crbr]



K = np.array([[406.952636, 0.000000, 366.184147], [0.000000, 405.671292, 244.705127], [0.000000, 0.000000, 1.000000]])
P = add_column(K, [[0], [0], [1]])



pts = [[camera_project(P, ltl), cltl],
       [camera_project(P, ltr), cltr],
       [camera_project(P, lbl), clbl],
       [camera_project(P, lbr), clbr]]


H = compute_homography(pts)


Composite = np.matmul(H, P)
print("Homography Matrix H:\n", H)
print()
print()
print("Projection Matrix P:\n", P)
print()
print()
print("Composite(H x P):\n" , Composite)


def main():
    image = cv2.imread("./image.png")
    output = camera_project(Composite, rtr)
    for point in world_pts: # Iterate through the world points, and use homography to project them
        draw_point(image, camera_project(Composite, point), "green")
    for point in original_pixel_pts: # Given locations of the world points in the image.
        draw_pt = (int(point[0]), int(point[1]))
        draw_point(image, draw_pt, "red")
    display_image(image)

main()

