cmake_minimum_required(VERSION 3.9)
project(caffe_crowd_test)

include_directories(/usr/local/include)
include_directories(/usr/local/cuda/include)
include_directories(/home/pengshanzhen/caffe/include)
include_directories(/home/pengshanzhen/caffe/build/src)

##
link_directories(/usr/local/lib)
link_directories(/home/pengshanzhen/caffe/build/lib)
link_directories(/usr/local/cuda/lib64)

#link_directories(/usr/local/ssl/lib)
link_directories(/usr/lib/x86_64-linux-gnu/hdf5/serial)
#link_directories(/usr/lib/)
#link_directories(/lib/x86_64-linux-gnu/)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
add_executable(caffe_crowd_test main.cpp CrowdEstimate.cpp)

target_link_libraries(caffe_crowd_test opencv_core opencv_imgproc opencv_highgui opencv_ml opencv_video opencv_features2d opencv_calib3d opencv_objdetect opencv_imgcodecs opencv_videoio
        opencv_flann boost_serialization boost_system boost_filesystem glog caffe hdf5 hdf5_hl boost_thread protobuf atlas cublas cudart culibos curand cudnn
        ssl3 pthread dl rt)