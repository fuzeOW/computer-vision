# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/project_wly_gpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/project_wly_gpu

# Include any dependencies generated for this target.
include CMakeFiles/project_wly_gpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/project_wly_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project_wly_gpu.dir/flags.make

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o: CMakeFiles/project_wly_gpu.dir/flags.make
CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o: project_wly.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/project_wly_gpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o -c /home/nvidia/project_wly_gpu/project_wly.cpp

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/project_wly_gpu.dir/project_wly.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/project_wly_gpu/project_wly.cpp > CMakeFiles/project_wly_gpu.dir/project_wly.cpp.i

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/project_wly_gpu.dir/project_wly.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/project_wly_gpu/project_wly.cpp -o CMakeFiles/project_wly_gpu.dir/project_wly.cpp.s

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.requires:

.PHONY : CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.requires

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.provides: CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.requires
	$(MAKE) -f CMakeFiles/project_wly_gpu.dir/build.make CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.provides.build
.PHONY : CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.provides

CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.provides.build: CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o


# Object files for target project_wly_gpu
project_wly_gpu_OBJECTS = \
"CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o"

# External object files for target project_wly_gpu
project_wly_gpu_EXTERNAL_OBJECTS =

project_wly_gpu: CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o
project_wly_gpu: CMakeFiles/project_wly_gpu.dir/build.make
project_wly_gpu: /home/nvidia/caffe/build/lib/libcaffe.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_highgui.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_core.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_imgproc.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_imgcodecs.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_videoio.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_features2d.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_video.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_flann.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_calib3d.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudafeatures2d.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudabgsegm.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudaimgproc.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudaarithm.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudafilters.so
project_wly_gpu: /home/nvidia/opencv-3.1.0/build/lib/libopencv_cudawarping.so
project_wly_gpu: /usr/lib/aarch64-linux-gnu/libglog.so
project_wly_gpu: /usr/lib/aarch64-linux-gnu/libboost_system.so
project_wly_gpu: CMakeFiles/project_wly_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/project_wly_gpu/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable project_wly_gpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project_wly_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project_wly_gpu.dir/build: project_wly_gpu

.PHONY : CMakeFiles/project_wly_gpu.dir/build

CMakeFiles/project_wly_gpu.dir/requires: CMakeFiles/project_wly_gpu.dir/project_wly.cpp.o.requires

.PHONY : CMakeFiles/project_wly_gpu.dir/requires

CMakeFiles/project_wly_gpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project_wly_gpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project_wly_gpu.dir/clean

CMakeFiles/project_wly_gpu.dir/depend:
	cd /home/nvidia/project_wly_gpu && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/project_wly_gpu /home/nvidia/project_wly_gpu /home/nvidia/project_wly_gpu /home/nvidia/project_wly_gpu /home/nvidia/project_wly_gpu/CMakeFiles/project_wly_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/project_wly_gpu.dir/depend

