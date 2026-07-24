cmake_minimum_required(VERSION 3.10)
project(bev_mono_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

option(ENABLE_CUDA "Enable CUDA toolkit checks and compile definitions" ON)
option(ENABLE_CUDNN "Enable cuDNN checks and compile definitions" ON)
option(BUILD_TESTS "Build the Catch2 unit test suite" ON)

find_package(PkgConfig REQUIRED)

# Prefer custom /opt OpenCV when available, then fall back to system pkg-config paths.
set(BEV_OPENCV_PREFIX "/opt/opencv/4.12.0" CACHE PATH "Preferred OpenCV prefix")
if(EXISTS "${BEV_OPENCV_PREFIX}/lib/pkgconfig")
    set(ENV{PKG_CONFIG_PATH} "${BEV_OPENCV_PREFIX}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(OPENCV QUIET opencv)
if(NOT OPENCV_FOUND)
    pkg_check_modules(OPENCV REQUIRED opencv4)
endif()

set(BEV_OPENCV_VERSION ${OPENCV_VERSION})
set(BEV_OPENCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
set(BEV_OPENCV_LIBRARY_DIRS ${OPENCV_LIBRARY_DIRS})
set(BEV_OPENCV_LIBS ${OPENCV_LIBRARIES})

if(BEV_OPENCV_VERSION VERSION_LESS "4.10.0")
    message(FATAL_ERROR "OpenCV >= 4.10.0 is required. Found: ${BEV_OPENCV_VERSION}")
endif()

find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(ENABLE_CUDNN)
    find_path(CUDNN_INCLUDE_DIR cudnn.h)
    find_library(CUDNN_LIBRARY cudnn)
    if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
        message(FATAL_ERROR "ENABLE_CUDNN=ON but cuDNN was not found")
    endif()
endif()

add_library(bev_core INTERFACE)
target_include_directories(bev_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(bev_core INTERFACE ${BEV_OPENCV_INCLUDE_DIRS})
target_link_directories(bev_core INTERFACE ${BEV_OPENCV_LIBRARY_DIRS})
target_link_libraries(bev_core INTERFACE ${BEV_OPENCV_LIBS})
target_link_libraries(bev_core INTERFACE yaml-cpp::yaml-cpp Ceres::ceres)

if(BEV_OPENCV_PREFIX)
    set(CMAKE_BUILD_RPATH "${BEV_OPENCV_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH "${BEV_OPENCV_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

if(ENABLE_CUDNN)
    target_include_directories(bev_core INTERFACE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(bev_core INTERFACE ${CUDNN_LIBRARY})
    target_compile_definitions(bev_core INTERFACE BEV_HAS_CUDNN=1)
endif()

if(ENABLE_CUDA)
    target_link_libraries(bev_core INTERFACE CUDA::cudart)
    target_compile_definitions(bev_core INTERFACE BEV_HAS_CUDA=1)
endif()

find_package(Threads REQUIRED)

add_library(bev_radarcam STATIC
    src/radarcam/socketcan_reader.cpp
)
target_link_libraries(bev_radarcam PUBLIC bev_core Threads::Threads)

add_executable(calibration_tool_qt
    apps/calibration_tool_qt.cpp
    resources/calibration_tool.qrc
)
target_link_libraries(calibration_tool_qt PRIVATE bev_core bev_radarcam Qt5::Widgets)

add_executable(bev_runner apps/bev_runner.cpp)
target_link_libraries(bev_runner PRIVATE bev_core)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "Configured ${PROJECT_NAME}")
message(STATUS "  C++ Standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV       : ${BEV_OPENCV_VERSION}")
message(STATUS "  OpenCV libs  : ${BEV_OPENCV_LIBRARY_DIRS}")
message(STATUS "  Qt5 Widgets  : enabled")
message(STATUS "  CUDA         : ${ENABLE_CUDA}")
message(STATUS "  cuDNN        : ${ENABLE_CUDNN}")
message(STATUS "  yaml-cpp     : ${yaml-cpp_VERSION}")
message(STATUS "  Ceres        : ${Ceres_VERSION}")
message(STATUS "  BUILD_TESTS  : ${BUILD_TESTS}")
