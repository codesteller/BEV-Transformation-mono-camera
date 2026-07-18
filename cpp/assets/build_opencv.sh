#!/bin/bash
open_cv_version=4.12.0

cv2_dir=${HOME}/Apsoft/cv2/
mkdir -p "$cv2_dir"
cd "$cv2_dir"

cv2_install_dir="/opt/opencv/${open_cv_version}"

# # Prep the system
# sudo apt update && sudo apt upgrade -y
# sudo apt -y install build-essential checkinstall cmake pkg-config yasm
# sudo apt -y install git gfortran
# sudo apt -y install libjpeg8-dev libpng-dev libtiff5-dev
# sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-dev
# sudo apt -y install libxine2-dev libv4l-dev
# sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
# sudo apt -y install libgtk2.0-dev libtbb-dev libatlas-base-dev
# sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
# sudo apt -y install libvorbis-dev libx264-dev libopencore-amrnb-dev libopencore-amrwb-dev  
# sudo apt -y install x264 v4l-utils
# sudo apt -y install libprotobuf-dev protobuf-compiler
# sudo apt -y install libgoogle-glog-dev libgflags-dev
# sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen flake8 pylint
# sudo apt -y install gcc-12 g++-12


# # Git clone OpenCV and OpenCV contrib repositories
# git clone --branch ${open_cv_version} https://github.com/opencv/opencv.git
# git clone --branch ${open_cv_version} https://github.com/opencv/opencv_contrib.git

# Build OpenCV
cd opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=${cv2_install_dir} \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D CUDA_ARCH_BIN="8.7" \
      -D CUDA_ARCH_PTX="" \
      -D OPENCV_DNN_CUDA=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=ON \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D BUILD_opencv_cudacodec=ON \
      -D WITH_NVCUVID=OFF \
      -D WITH_NVCUVENC=OFF \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_GTK=ON \
      -D WITH_GTK3=ON \
      -D WITH_GTK_2_X=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_PC_FILE_NAME=opencv.pc \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")  \
      -D PYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
      -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=$(which python3) \
      -D CUDA_HOST_COMPILER=/usr/bin/g++-12 \
      -D BUILD_EXAMPLES=ON ..
    
make -j$(nproc)
sudo make install
sudo ldconfig

# Update the library path  & Binary path
echo " -------- OpenCV ${open_cv_version} installed successfully at ${cv2_install_dir} --------"
echo " ---- OpenCV Path and Environment variables ----" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=${cv2_install_dir}/lib/pkgconfig:\$PKG_CONFIG_PATH" >> ~/.bashrc
echo "export PATH=${cv2_install_dir}/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${cv2_install_dir}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Verify the installation
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Find CMake configuration file and add it to the CMake module path
if [ -f "${cv2_install_dir}/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    echo "OpenCVConfig.cmake found at ${cv2_install_dir}/lib/cmake/opencv4"
    echo "export CMAKE_PREFIX_PATH=${cv2_install_dir}/lib/cmake/opencv4:\$CMAKE_PREFIX_PATH" >> ~/.bashrc
    source ~/.bashrc
else
    echo "OpenCVConfig.cmake not found. Please check the installation."
fi




