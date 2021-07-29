cmake_minimum_required(VERSION 3.6.2)

project(SimplexPP)

add_executable(SimplexPP simplex.cpp)


# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)