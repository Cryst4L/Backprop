cmake_minimum_required(VERSION 3.1)

project(Backprop)

# Check the OS -----------------------------------------------------------------
if(NOT UNIX)
	message(FATAL_ERROR " Only Unix based systems are supported")
endif()

# Check the CMake usage --------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR " Inappropriate CMake usage. \n" 
	                    " Should be called in a build directory: \n" 
	                    " $ mkdir build; cmake ..")
endif()

# Find Eigen3 ------------------------------------------------------------------
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
	message(FATAL_ERROR " Could not locate the Eigen3 library. \n"
	                    " Please ensure it is installed on your machine.")
endif()

# Include Backprop and Eigen3 --------------------------------------------------
include_directories(. ${EIGEN3_INCLUDE_DIRS})

# Set the compiler flags -------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -s -Wall -Wextra")
endif()

# Build the sources ------------------------------------------------------------
#file(GLOB_RECURSE BACKPROP_SRC "src/*.cpp")
#add_library(Backprop SHARED ${BACKPROP_SRC})

# Find GNUPlot -----------------------------------------------------------------
find_program(GNUPLOT_EXECUTABLE gnuplot)
if (NOT GNUPLOT_EXECUTABLE)
	message(WARNING "GNUPlot is required to run the example.")
else()

# Build the example ------------------------------------------------------------
option(BUILD_EXAMPLE "Build the example." ON)
if(BUILD_EXAMPLE)
	add_executable(EX1 examples/ex1.cpp)
	add_executable(EX2 examples/ex2.cpp)
	add_executable(EX3 examples/ex3.cpp)
	#target_link_libraries(EXAMPLE Backprop)
	execute_process(COMMAND ${CMAKE_SOURCE_DIR}/data/MNIST/get_mnist.py
	                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/data/MNIST)
	endif()
endif()

# Copy the dataset -----------------------------------------------------------
file(COPY "data" DESTINATION ".")




# TODO: subfolder for the example, the python script with image? NO
# TODO: Doxygen?

