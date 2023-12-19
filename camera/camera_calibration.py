import numpy as np
import cv2
import glob
import yaml
import os
from datetime import datetime
import tqdm
from . import calibrator


# Supported calibration patterns
class Patterns:
    Chessboard, Circles, ACircles, ChArUco = list(range(4))


class CalibrationException(Exception):
    pass


class Calibration:
    def __init__(
        self,
        images_dir="images/calib_images",
        calibration_file="calibration_matrix.yaml",
        chessboard_size=(8, 6),
        chessboard_square_size=0.047,
        images_dims=(1920, 1020),
        record=True,
        visualize=False,
        camera_model="pinhole",
        camera_name="usbcam",
    ):
        self.images_dir = images_dir
        self.chessboard_size = (8, 6)
        self.calibration_file = calibration_file
        self.record = record
        self.images_path = list()
        self.chessboard_square_size = chessboard_square_size
        self.images_dims = images_dims
        self.visualize = visualize
        self.alpha = 0.0
        self.rotation = np.eye(3, dtype=np.float64)
        self.projection = np.zeros((3, 4), dtype=np.float64)
        self.camera_model = camera_model
        self.camera_name = camera_name

    def run_capture(self):
        os.makedirs(self.images_dir, exist_ok=True)
        if self.record == True:
            os.system("rm -rf {}/*".format(self.images_dir))
            self.camera_capture()

        images = []
        for image in os.listdir(self.images_dir):
            if image.endswith(".png"):
                images.append(os.path.join(self.images_dir, image))

        self.images_path = images

    def camera_capture(self):
        cap = cv2.VideoCapture(4)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.images_dims[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.images_dims[1])
        count = 0
        while True:
            ret, frame = cap.read()
            # cv2.imshow("frame", frame)

            # resize the frame, convert it to grayscale
            gray_frame_resized = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (int(self.images_dims[0] / 4), int(self.images_dims[1] / 4)),
            )
            ret, corners = cv2.findChessboardCorners(
                gray_frame_resized, self.chessboard_size, None
            )
            gray_frame = cv2.drawChessboardCorners(
                gray_frame_resized, self.chessboard_size, corners, ret
            )
            cv2.imshow("Detections", gray_frame_resized)

            key_val = cv2.waitKey(1) & 0xFF

            if key_val == ord("c"):
                filename = os.path.join(self.images_dir, "image{}.png".format(count))
                cv2.imwrite(filename, frame)
                current_time = datetime.now().strftime("%H:%M:%S")
                print(
                    "[INFO : {}] ({}).Image Captured {}".format(
                        current_time, count, filename
                    )
                )

                count += 1
            elif key_val == ord("q"):
                break
            else:
                pass

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _get_dist_model(dist_params, cam_model):
        if cam_model == "pinhole":
            if dist_params is None:
                return "plumb_bob"
            else:
                if len(dist_params) == 4:
                    return "plumb_bob"
                elif len(dist_params) == 5:
                    return "rational_polynomial"
                elif len(dist_params) == 8:
                    return "thin_prism_model"
                else:
                    raise CalibrationException(
                        "Invalid number of distortion parameters"
                    )
        elif cam_model == "fisheye":
            if dist_params is None:
                return "equidistant"
            else:
                if len(dist_params) == 4:
                    return "equidistant"
                else:
                    raise CalibrationException(
                        "Invalid number of distortion parameters"
                    )
        else:
            raise CalibrationException("Invalid camera model")

    def _save_calibration(self, calib_file, data):

        # d, k, r, p, size, cam_model
        d = np.array(data["dist_coeff"])
        k = np.array(data["camera_matrix"])
        r = np.array(data["rotation"])
        p = np.array(data["projection"])
        size = (data["image_width"], data["image_height"])
        cam_model = data["camera_model"]
        name = data["camera_name"]
        dist_model = "plumb_bob"


        # Check all data correct
        calmessage = "\n".join([
            "image_width: %d" % size[0],
            "image_height: %d" % size[1],
            "camera_name: " + name,
            "camera_matrix:",
            "  rows: 3",
            "  cols: 3",
            "  data: " + self._format_mat(k, 5),
            "distortion_model: " + dist_model,
            "distortion_coefficients:",
            "  rows: 1",
            "  cols: %d" % d.size,
            "  data: [%s]" % ", ".join("%8f" % x for x in d.flat),
            "rectification_matrix:",
            "  rows: 3",
            "  cols: 3",
            "  data: " + self._format_mat(r, 8),
            "projection_matrix:",
            "  rows: 3",
            "  cols: 4",
            "  data: " + self._format_mat(p, 5),
            ""
        ])

        with open(calib_file, "w") as f:
            f.write(calmessage)
    
    @staticmethod
    def _format_mat(x, precision):
            return ("[%s]" % (
                np.array2string(x, precision=precision, suppress_small=True, separator=", ")
                    .replace("[", "").replace("]", "").replace("\n", "\n        ")
            ))

    def calibrate(self):
        images = glob.glob(self.images_dir + "/*.png")
        if len(images) == 0:
            print("No images found in {}".format(self.images_dir))
            return

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = (
            np.mgrid[
                0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
            ].T.reshape(-1, 2)
            * self.chessboard_square_size
        )
        objpoints = []
        imgpoints = []

        found = 0
        for fname in tqdm.tqdm(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            cv2.destroyAllWindows()
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                img = cv2.drawChessboardCorners(
                    img, self.chessboard_size, corners2, ret
                )
                found += 1
                if self.visualize:
                    cv2.imshow("img", img)
                    cv2.waitKey(500)

        print("Number of images used for calibration: ", found)

        reproj_error, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, 0
        )
        ncm, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics, distortion, self.images_dims, self.alpha
        )
        for j in range(3):
            for i in range(3):
                self.projection[j, i] = ncm[j, i]

        mapx, mapy = cv2.initUndistortRectifyMap(
            intrinsics, distortion, self.rotation, ncm, self.images_dims, cv2.CV_32FC1
        )
        data = {
            "camera_model": self.camera_model,
            "camera_name": self.camera_name,
            "image_width": self.images_dims[0],
            "image_height": self.images_dims[1],
            "camera_matrix": np.asarray(intrinsics).tolist(),
            "dist_coeff": np.asarray(distortion).tolist(),
            "rotation": np.asarray(self.rotation).tolist(),
            "projection": np.asarray(self.projection).tolist(),
            "mapx": mapx.tolist(),
            "mapy": mapy.tolist(),
            "rvects": np.asarray(rvecs).tolist(),
            "tvects": np.asarray(tvecs).tolist(),
        }

        self._save_calibration(self.calibration_file, data)

        return intrinsics, distortion
