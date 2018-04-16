cmake_minimum_required(VERSION 3.5)
project(T3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -O3 -std=c++14 -D ILEASILE")
set(SOURCE_FILES main.cpp Timer.h)
set(CMAKE_CXX_COMPILER g++)

add_executable(T3 ${SOURCE_FILES})
