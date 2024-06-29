cmake_minimum_required(VERSION 3.10)
project(pycpptools)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)
add_executable(main src/main.cpp src/utils.cpp)