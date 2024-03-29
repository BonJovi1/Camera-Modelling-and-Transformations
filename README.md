# Camera-Modelling-and-Transformations

## LiDAR Camera Fusion 

Open folder q1, and run `jupyter-notebook Q1.ipynb` OR `python3 q1.py`  
If the notebook doesn't load on GitHub, use the nbviewer:  
https://nbviewer.jupyter.org/github/BonJovi1/Camera-Modelling-and-Transformations/blob/master/q1/Q1.ipynb  

The question: 

You are working on a self-driving car and the team has decided to fuse image data from a camera with distance measurements from a LiDAR (a laser scanner with a 360 degree field-of-view that records distance measurements) in order to associate every point in the image with accurate distance measurements. A LiDAR frame and its corresponding camera image have been provided to you as lidar-points.bin and image.png respectively. The camera calibration matrix, K, is provided inside K.txt.

The LiDAR’s frame is defined such that its X-axis points forward, Y-axis points to the left, and its Z-axis points upwards. And the camera’s frame is defined such that its Z-axis points forward, X-axis points to the right, and Y-axis points downwards. The camera’s center is found to be (via extrinsic calibration) 8 cm below, 6 cm to the left, and 27 cm in front of the LiDAR’s center (as measured from the LiDAR). 

Both the sensors are positioned such that the camera’s Z-axis and the LiDAR’s X-axis are perfectly parallel. Compute the transformation (R,t) required to transform points in the LiDAR’s frame to the camera’s frame. Give the transformation in both (a) homogeneous matrix form, and (b) XYZ Euler angles (RPY)-translation form.

Then, using this computed transformation and the provided camera calibration matrix, project the LiDAR’s points onto the image plane and visualize it as shown below. The color code (colormap) corresponds to the depth of the points in the image (color is optional, but it helps in debugging). Use matplotlib or any equivalent library for plotting the points on the image. Code for loading the LiDAR points in Python is provided.

## 3D Bounding Box

Open folder q2, and run `jupyter-notebook Q2.ipynb` OR `python3 q2.py`  
If the notebook doesn't load on GitHub, use the nbviewer:  
https://nbviewer.jupyter.org/github/BonJovi1/Camera-Modelling-and-Transformations/blob/master/q2/Q2.ipynb  

The question:

You’ve been provided with an image, also taken from a self-driving car, that shows another car in front. The camera has been placed on top of the car, 1.65 m from the ground, and assume the image plane is perfectly perpendicular to the ground. K is provided to you. Your task is to draw a 3D-bounding box around the car in front as shown. Your approach should be to place eight points in the 3D world such that they surround all the corners of the car, then project them onto the image, and connect the projected image points using lines. You might have to apply a small 5à rotation about the vertical axis to align the box perfectly. Rough dimensions of the car - h: 1.38 m, w: 1.51, l: 4.10. 

## April Tags

Open folder q3, and run `jupyter-notebook Q3.ipynb` OR `python3 q3.py`  
If the notebook doesn't load on GitHub, use the nbviewer:  
https://nbviewer.jupyter.org/github/BonJovi1/Camera-Modelling-and-Transformations/blob/master/q3/Q3.ipynb

The question:

AprilTags are artificial landmarks that play an important role in robotics and augmented-reality. These are 2D QR-code like tags that are designed to be easily recognized. They are used in robotics for object recognition and accurate pose estimation where perception (from natural features) is not possible or is not the central focus. They are greatly robust to occlusion, warping, distortions, and lighting variations.
An image of a planar surface with two AprilTags stuck on it is provided. Your task is to estimate the pose of the camera using these tags.
You can do this by applying the Direct Linear Transform (DLT), like we saw in Zhang’s method for camera calibration. The dimensions of each tag and their corner pixel locations are provided in april tags info.txt. Using these locations and their corresponding world locations, estimate the 3 × 3 homography matrix, H, that maps these world points to their image locations. Verify the accuracy of your estimated H matrix by projecting the physical corner points (in world frame) onto the image plane, and visualize them along with the provided corner pixel locations.



