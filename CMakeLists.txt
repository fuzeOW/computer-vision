cmake_minimum_required (VERSION 2.8)  
  
project (project_wly_gpu)  
  
add_executable(project_wly_gpu project_wly.cpp)  
  
include_directories ( /home/nvidia/caffe/include  
    /usr/local/include  
    /usr/local/cuda/include  
    /usr/include )  
  
target_link_libraries(project_wly_gpu 
    /home/nvidia/caffe/build/lib/libcaffe.so  
    #/usr/lib/aarch64-linux-gnu/libopencv_highgui.so 
    #/usr/lib/aarch64-linux-gnu/libopencv_core.so  
    #/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so  
    #/usr/lib/aarch64-linux-gnu/libglog.so  
    #/usr/lib/aarch64-linux-gnu/libboost_system.so

    #/usr/local/lib/libopencv_highgui.so 
    #/usr/local/lib/libopencv_core.so
    #/usr/local/lib/libopencv_imgproc.so
    #/usr/local/lib/libopencv_imgcodecs.so   
    #/usr/lib/aarch64-linux-gnu/libglog.so 
    #/usr/lib/aarch64-linux-gnu/libboost_system.so

    /home/nvidia/opencv-3.1.0/build/lib/libopencv_highgui.so 
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_core.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_imgproc.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_imgcodecs.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_videoio.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_features2d.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_video.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_flann.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_calib3d.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudafeatures2d.so

    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudabgsegm.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudaimgproc.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudaarithm.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudafilters.so
    /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudawarping.so

    /usr/lib/aarch64-linux-gnu/libglog.so 
    /usr/lib/aarch64-linux-gnu/libboost_system.so 

    #/usr/lib/aarch64-linux-gnu/libv4l/v4l2convert.so
    )  
