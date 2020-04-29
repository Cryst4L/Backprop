cmake_minimum_required(VERSION 3.1)
project(Backprop)

# Check the OS -----------------------------------------------------------------
if (NOT UNIX)
	message(FATAL_ERROR " Only Unix based systems are supported")
endif()

# Ensure a proper CMake usage --------------------------------------------------
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR " Inappropriate CMake usage.\n"
	                    " Must be called in a 'build' direcory as follow:\n"
	                    " $ mkidr build && cmake ..")
endif()

# Find Eigen3 ------------------------------------------------------------------
find_package(Eigen3)
if(NOT EIGEN3_FOUND)
	message(FATAL_ERROR " Could not locate the Eigen3 library. \n"
	                    " Please ensure it is installed on your machine.")
endif()

# Find GNUPlot -----------------------------------------------------------------
find_program(GNUPLOT_EXECUTABLE gnuplot)
if (NOT GNUPLOT_EXECUTABLE)
		message(FATAL_ERROR " GNUPlot is required to diplay data. \n"
		                    " Please ensure it is installed on your machine.")	
endif()

# Include Backprop and Eigen3 --------------------------------------------------
include_directories(. ${EIGEN3_INCLUDE_DIRS})

# Add Backprop source files ----------------------------------------------------
file(GLOB_RECURSE BACKPROP_SRC "src/*.cpp")

# Set the compiler flags -------------------------------------------------------
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
endif()

# Make the library -------------------------------------------------------------
add_library(Backprop SHARED ${BACKPROP_SRC})

# Get MNIST dataset ------------------------------------------------------------
execute_process(COMMAND ${CMAKE_SOURCE_DIR}/data/get_mnist.py
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/data)	

# Build the example ------------------------------------------------------------
option(BUILD_EXAMPLE "Build the example." ON)
if (BUILD_EXAMPLE)
	add_executable(example example.cpp)
	target_link_libraries(example Backprop)
endif()
