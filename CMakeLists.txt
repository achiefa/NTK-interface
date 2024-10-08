cmake_minimum_required (VERSION 3.5)

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()

project(NTK)
option(NTK_ENABLE_TESTING "Build tests" OFF)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
  set(NTK_ENABLE_TESTING ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# CXX standard for the project
set(NTK 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Determine the compiler for the current system
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  add_compile_options(-O3 -Wall -fPIC -lpthread -fopenmp) #linux
else()
  add_compile_options(-O3 -stdlib=libc++ -Wall -fPIC) #mac
endif()

# Set flags for gcc-like compilers
set (gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")

configure_file(
    "${PROJECT_SOURCE_DIR}/inc/NTK/version.h.in"
    "${PROJECT_SOURCE_DIR}/inc/NTK/version.h"
)

include_directories(${PROJECT_SOURCE_DIR}/inc)

#set(CMAKE_MODULE_PATH "")
find_package(PkgConfig REQUIRED)
find_package(benchmark REQUIRED)
pkg_search_module(EIGEN3 REQUIRED eigen3)
pkg_search_module(YAML REQUIRED yaml-cpp)
find_package(ceres REQUIRED)
#find_package (glog 0.6.0 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# NNAD
find_program(NNAD_CONFIG nnad-config REQUIRED)
if (NNAD_CONFIG)
    exec_program(${NNAD_CONFIG}
        ARGS --cflags
        OUTPUT_VARIABLE NNAD_CFLAGS
    )
    set(NNAD_CFLAGS ${NNAD_CFLAGS} CACHE STRING INTERNAL)
endif(NNAD_CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NNAD_CFLAGS} ${EIGEN3_CFLAGS} ${Ceres_CFLAGS} ${YAML_CFLAGS}")

# build target library
add_subdirectory(src bin)
add_subdirectory(run)

if(NTK_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()