cmake_minimum_required(VERSION 3.1)
project(Backprop)
# Check the OS -------------------------------------------------------
if (NOT UNIX)
	message(FATAL_ERROR " Only Unix based systems are supported")
endif()
# Ensure a proper CMake usage ----------------------------------------
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR " Inappropriate CMake usage.\n"
	                    " Must be called in a 'build' direcory as follow:\n"
	                    " $ mkidr build && cmake ..")
endif()
# Set the compiler flags ---------------------------------------------
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")
endif()
# Include the headers ------------------------------------------------
include_directories(.)
# Add the source files -----------------------------------------------
file(GLOB_RECURSE BACKPROP_SRC "src/*.cpp")
# Make the library ---------------------------------------------------
add_library(Backprop SHARED ${BACKPROP_SRC})
# Build the example --------------------------------------------------
option(BUILD_EXAMPLE "Build the example." ON)
if (BUILD_EXAMPLE)
	add_executable(example example.cpp)
	target_link_libraries(example Backprop)
endif()
