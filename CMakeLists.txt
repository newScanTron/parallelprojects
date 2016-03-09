cmake_minimum_required(VERSION 3.3)
project(parallel)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES
    helloworld.1.c
    helloworld.c
    linked.c
    linkedTasks.c
    mandel.c
    serialPi.c)

add_executable(parallel ${SOURCE_FILES})