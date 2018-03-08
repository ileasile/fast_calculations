cmake_minimum_required(VERSION 3.5)
project(T2)

set(CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp")
set(SOURCE_FILES main.cpp)
# set(SOURCE_FILES main_old.cpp)
add_executable(T2 ${SOURCE_FILES})
