cmake_minimum_required(VERSION 2.8.3)
project(pycpptools)

set(CMAKE_BUILD_TYPE "Release")
# explicitly set std=c++14 to remove errors from pcl library
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(cpp/include ${catkin_INCLUDE_DIRS})

#############
# LIBRARIES #
#############
# cs_add_library(${PROJECT_NAME})

##############
# EXECUTABLE #
##############

cs_add_executable(pycpptools_hello_world cpp/src/main.cpp cpp/src/utils.cpp)
# target_link_libraries(main ${PROJECT_NAME})

########
# TEST #
########
# 

##########
# EXPORT #
##########
cs_install()
cs_export()
