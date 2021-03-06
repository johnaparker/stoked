cmake_minimum_required(VERSION 3.1)
project(cpp)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -pedantic -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx -flto -fPIC")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#find_package(pybind11 REQUIRED)
add_subdirectory(pybind11)

include_directories(src)

set(SOURCES ${SOURCES} 
    src/solver.cpp
)

set(BINDINGS
    src/main_pyb.cpp
    src/solver_pyb.cpp
)

add_library(cpp MODULE "${SOURCES}" "${BINDINGS}")
target_link_libraries(cpp PRIVATE pybind11::module Eigen3::Eigen)
set_target_properties(cpp PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")

#if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    #add_executable(debug test/debug.cpp "${SOURCES}")
    #set_target_properties(debug PROPERTIES COMPILE_FLAGS "-w")
    #target_link_libraries(debug cpp pybind11::embed)
#endif()
