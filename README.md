# BEV-Transformation-mono-camera
It converts a video input to BEV representation for depth estimation and object detection. The repository can be used for The entire calibration process of monocular camera to BEV representation. 

## Features:
- [ ] Monocular camera intrinsics calibration based on OpenCV
    - [ ] UI based on QT to do the calibration like the OpenCV calibration tool on ROS with Skew, Scale, X, Y.
    - [ ] Save the calibration result to a YAML file.
    - [ ] Once Calibrated use intrinsics to undistort the image.
- [ ] Monocular camera extrinsics calibration based on OpenCV 4-Point Homography transformation.
    - [ ] UI based on same QT window to do the calibration with View Finder detecting 4 aruco markers on the ground plane and giving them each an ID.
    - [ ] On one side (left) of the window, put text boxes to enter the real-world distances between the markers in meters (height and width) and the distance from the camera to the ground plane in meters.
    - [ ] Save the calibration result to a YAML file.
    - [ ] Once Calibrated use extrinsics to transform the image to BEV transformation.
    - [ ] Use the BEV transformation to do depth estimation and object detection.
    

