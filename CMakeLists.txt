cmake_minimum_required (VERSION 3.5)

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()

project(NTK)

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

# Set optimisation flags
#option(OPTIMISE "Build shared-libray instead of static-libray" ON)
#if(OPTIMISE EQUAL ON)
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#endif(OPTIMISE)

configure_file(
    "${PROJECT_SOURCE_DIR}/inc/NTK/version.h.in"
    "${PROJECT_SOURCE_DIR}/inc/NTK/version.h"
)

include_directories(${PROJECT_SOURCE_DIR}/inc)

find_package(PkgConfig REQUIRED)
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NNAD_CFLAGS} ${EIGEN3_CFLAGS} ${Ceres_CFLAGS} ${YAML_CFLAGS} ")

# build target library
add_subdirectory(src)
add_subdirectory(run)
add_subdirectory(test)

# Installing and Testing
# ______________________________________________________________________________________
# Configuration script
##set(prefix ${CMAKE_INSTALL_PREFIX})
##set(exec_prefix "${prefix}")
##set(includedir "${prefix}/include")
##set(libdir "${prefix}/lib")
##install (TARGETS train DESTINATION bin)

# Testing sessione
#_____________________________________________________
#enable_testing()
#set (TEST_RUNCARD "${PROJECT_SOURCE_DIR}/tests/test_runcard.yaml")
#add_test(NAME Test COMMAND train TEST_RUNCARD)