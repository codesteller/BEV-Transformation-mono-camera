import sys
from matplotlib import pyplot
import cv2
import numpy as np
import yaml
import os
import math


PI = 3.14159265359

class Rectification:
    def __init__(self, calib_file, images_dir="images"):
        self.calib_file = calib_file
        self.camera_matrix = None
        self.dist_coeff = None
        self.images_dir = images_dir
        self.image_height = 1020
        self.image_width = 1920
        self.check_calib_file()

    def check_calib_file(self):
        if not os.path.isfile(self.calib_file):
            print("[ERROR] Calibration file not found {}".format(self.calib_file))
            sys.exit(1)

    # def load_calibration_cv2(self, ):

    def load_calibration(self):
        with open(self.calib_file) as f:
            loadeddict = yaml.load(f, Loader=yaml.FullLoader)
            camera_matrix_raw = loadeddict.get("camera_matrix")
            dist_coeff_raw = loadeddict.get("distortion_coefficients")

            # convert list to numpy array reshape camera matrix
            self.camera_matrix = np.array(camera_matrix_raw["data"], dtype=np.float32)
            self.camera_matrix = self.camera_matrix.reshape(
                (camera_matrix_raw["rows"], camera_matrix_raw["cols"])
            )

            # convert list to numpy array reshape distortion matrix
            self.dist_coeff = np.array(dist_coeff_raw["data"])
            self.dist_coeff = self.dist_coeff.reshape(
                (dist_coeff_raw["rows"], dist_coeff_raw["cols"])
            )

            self.image_height = int(loadeddict.get("image_height"))
            self.image_width = int(loadeddict.get("image_width"))

    def undistort(self, image, fisheye=False):
        if fisheye:
            h, w = image.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeff, (w, h), 1, (w, h)
            )
            return cv2.undistort(
                image, self.camera_matrix, self.dist_coeff, None, newcameramtx
            )
        else:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeff, None, None)

    def undistort_images(self, write=False):
        images = []
        for image in os.listdir(self.images_dir):
            if image.endswith(".png"):
                images.append(os.path.join(self.images_dir, image))

        for image in images:
            im = cv2.imread(image)
            im_ = self.undistort(im)
            if write:
                cv2.imwrite(image, im)
                print("[INFO] Undistorted Image Saved {}".format(image))
            else:
                cv2.imshow("Original Image", im)
                cv2.imshow("Rectified Image", im_)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    def undistort_frame(self, im):
        im_ = self.undistort(im)
        return im_
    
    def update_perspective(self, val):
        alpha = (cv2.getTrackbarPos("Alpha", "Result") - 90) * PI / 180
        beta = (cv2.getTrackbarPos("Beta", "Result") - 90) * PI / 180
        gamma = (cv2.getTrackbarPos("Gamma", "Result") - 90) * PI / 180
        focalLength = cv2.getTrackbarPos("f", "Result")
        dist = cv2.getTrackbarPos("Distance", "Result")

        w, h = (self.image_width, self.image_height)

        A1 = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 0],
                    [0, 0, 1]], dtype=np.float32)

        RX = np.array([[1, 0, 0, 0],
                    [0, math.cos(alpha), -math.sin(alpha), 0],
                    [0, math.sin(alpha), math.cos(alpha), 0],
                    [0, 0, 0, 1]], dtype=np.float32)

        RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                    [0, 1, 0, 0],
                    [math.sin(beta), 0, math.cos(beta), 0],
                    [0, 0, 0, 1]], dtype=np.float32)

        RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
                    [math.sin(gamma), math.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)

        R = np.dot(np.dot(RX, RY), RZ)

        T = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dist],
                    [0, 0, 0, 1]], dtype=np.float32)

        K = np.array([[focalLength, 0, w / 2, 0],
                    [0, focalLength, h / 2, 0],
                    [0, 0, 1, 0]], dtype=np.float32)

        transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)

        ret, frame = capture.read()
        if not ret:
            return

    destination = cv2.warpPerspective(frame, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    cv2.imshow("Result", destination)

    def undistort_camera(self, camera_device=0):
        dev = "/dev/video{}".format(camera_device)
        cap = cv2.VideoCapture(dev)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame1 = self.undistort(frame)
            cv2.imshow("Original frame", frame)
            cv2.imshow("Undistorted frame", frame1)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rectify = Rectification(
        calib_file="camera_calibration/configs/BR01FU9650-1920x1020.yaml",
        images_dir="camera_calibration/images/scooter_images",
    )
    rectify.load_calibration()
    # rectify.undistort_images()
    rectify.undistort_camera(4)
