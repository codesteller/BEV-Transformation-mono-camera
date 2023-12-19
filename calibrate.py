"""
 # @ Author: Pallab Maji
 # @ Create Time: 2023-12-07 13:26:42
 # @ Modified time: 2023-12-07 13:40:54
 # @ Description: Enter description here
 """

import camera.camera_calibration as camera_calibration
import camera.camera_rectification as camera_rectification
import cv2
import numpy as np
import math
import os


# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")


def main():
    # Enter the Parameters
    camera_name = "usbcam_econ"
    camera_device = 4
    camera_model = "pinhole"
    images_dir = "assets/images/calib_images"
    calibration_file = "assets/calibration_matrix_usbcam-econ.yaml"
    chessboard_size = (8, 6)
    chessboard_square_size = 0.035
    images_dims = (1920, 1080)
    do_calibration = False

    if do_calibration:
        calib = camera_calibration.Calibration(
            images_dir=images_dir,
            calibration_file=calibration_file,
            chessboard_size=chessboard_size,
            chessboard_square_size=chessboard_square_size,
            images_dims=images_dims,
            record=False,
            visualize=False,
            camera_model=camera_model,
            camera_name=camera_name,
        )
        calib.run_capture()
        calib.calibrate()

    rectify = camera_rectification.Rectification(
        calib_file=calibration_file, images_dir=images_dir
    )
    rectify.load_calibration()
    
    rectify.undistort_camera(camera_device=camera_device)






if __name__ == "__main__":
    main()
