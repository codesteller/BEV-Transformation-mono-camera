/**
 * @ Author: Pallab Maji
 * @ Create Time: 2023-12-05 14:52:54
 * @ Modified time: 2023-12-05 16:07:41
 * @ Description: Enter description here
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <yaml-cpp/yaml.h>

struct calib_params {
    int image_width;
    int image_height;
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

struct calib_params get_calibration_params(std::string config_file) {

    YAML::Node config = YAML::LoadFile(config_file);
    
    calib_params calib;
    calib.image_width = config["image_width"].as<int>();
    calib.image_height = config["image_height"].as<int>();
    calib.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    calib.camera_matrix.at<double>(0, 0) = config["camera_matrix"]["data"][0].as<double>();
    calib.camera_matrix.at<double>(0, 1) = config["camera_matrix"]["data"][1].as<double>();
    calib.camera_matrix.at<double>(0, 2) = config["camera_matrix"]["data"][2].as<double>();
    calib.camera_matrix.at<double>(1, 0) = config["camera_matrix"]["data"][3].as<double>();
    calib.camera_matrix.at<double>(1, 1) = config["camera_matrix"]["data"][4].as<double>();
    calib.camera_matrix.at<double>(1, 2) = config["camera_matrix"]["data"][5].as<double>();
    calib.camera_matrix.at<double>(2, 0) = config["camera_matrix"]["data"][6].as<double>();
    calib.camera_matrix.at<double>(2, 1) = config["camera_matrix"]["data"][7].as<double>();
    calib.camera_matrix.at<double>(2, 2) = config["camera_matrix"]["data"][8].as<double>();
    calib.dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    calib.dist_coeffs.at<double>(0, 0) = config["distortion_coefficients"]["data"][0].as<double>();
    calib.dist_coeffs.at<double>(1, 0) = config["distortion_coefficients"]["data"][1].as<double>();
    calib.dist_coeffs.at<double>(2, 0) = config["distortion_coefficients"]["data"][2].as<double>();
    calib.dist_coeffs.at<double>(3, 0) = config["distortion_coefficients"]["data"][3].as<double>();
    calib.dist_coeffs.at<double>(4, 0) = config["distortion_coefficients"]["data"][4].as<double>();
    
    return calib;
}

cv::Mat undistort_image(std::string image_file, calib_params calib, bool show_image=false) {
    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    cv::Mat image_undistorted;
    cv::undistort(image, image_undistorted, calib.camera_matrix, calib.dist_coeffs);
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return image;
    }

    if (show_image){
        cv::imshow("Original Image", image);
        cv::imshow("Undistorted Image", image_undistorted);
        cv::waitKey(0);
    }
    
    return image_undistorted;
}

int main()
{
    
    calib_params calib;

    // Read the calibration parameters
    std::string config_file = "../test_images/BR01FU9650-1920x1020.yaml";
    calib = get_calibration_params(config_file);
    
    // Read the image
    std::string image_file = "../test_images/images-1920x1020/2.png";
    cv::Mat image_undistorted = undistort_image(image_file, calib, true);
    
    return 0;
    
}

