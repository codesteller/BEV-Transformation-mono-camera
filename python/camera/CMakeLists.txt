cmake_minimum_required(VERSION 3.20)
project(calibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Opencv
set(OpenCV_DIR /home/pallab/Apsoft/dev-utils/cv2/installation/OpenCV-4.8.1/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# YAML-CPP
set(YAML-CPP_DIR /home/pallab/Apsoft/dev-utils/yaml-parser/yaml-cpp/install/lib/cmake/yaml-cpp/)
find_package(YAML-CPP REQUIRED)
message(STATUS "YAML-CPP Libs: ${YAML_CPP_LIBRARIES}\n")

add_executable(${PROJECT_NAME} rectify.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${YAML_CPP_LIBRARIES})
