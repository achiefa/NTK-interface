cmake_minimum_required (VERSION 3.21)
project(NTK)

# Do not allow in-source builds
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed")
endif()

#------[ Query environment infomration ] -----
cmake_host_system_information(RESULT CORES QUERY NUMBER_OF_LOGICAL_CORES NUMBER_OF_PHYSICAL_CORES IS_64BIT)
list (GET CORES 0 LOGIC_CORES)
list (GET CORES 1 PHYS_CORES)
list (GET CORES 2 IS64_CORES)
message(STATUS "Logical cores: ${LOGIC_CORES}")
message(STATUS "Physical cores: ${PHYS_CORES}")
message(STATUS "Is 64 bit core: ${IS64_CORES}")

#]-----------------------[ CMake Config ]-----------------------
message(STATUS "Using CMake version ${CMAKE_VERSION}")

# Setting paths as global variables
set(PATH_TEST ${CMAKE_SOURCE_DIR}/test)
set(PATH_STD_TEST ${PATH_TEST}/std_test)

# [ Language settings ]
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#[=[ 
  Needed to avoid compiler enhancements that would
  take over that C++ standard library. In other words,
  the enchancements would be compiler-dependent and
  may lead to problems if the user swithces to other
  platforms.
# ]=]
set(CMAKE_CXX_EXTENSIONS OFF)

# Std library test
message(STATUS "Compiling and running test for std library...")
try_run(run_result compile_result
        SOURCES ${PATH_STD_TEST}/main.cpp 
        RUN_OUTPUT_VARIABLE output
        LOG_DESCRIPTION "Compiling and running std test."
        )
message(VERBOSE "   Run results: ${run_result}")
message(VERBOSE "   Compile results: ${compile_result}")
message(VERBOSE "   Output: ${output}")
if((run_result) AND (NOT compile_result))
  message(SEND_ERROR "Compiler could not compile and run the test file.")
elseif(run_result AND compile_result)
  message("     Done!")
endif()

# [ Interprocedural optimisation ]
# Check if the compiler supports interprocedural optimisation.
# This allows optimisation after linking (link tim  optimisation),
# so that all compilation units can be optimised as a single module.
include(CheckIPOSupported) 
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION True)
  message(STATUS "Interprocedural optimisation: ${ipo_supported}.")
else()
  message(STATUS "Interprocedural optimisation is not supported by your compiler: ${CMAKE_CXX_COMPILER}")
endif()

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()



# CXX standard for the project
set(NTK 1.0.0)

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

# Installing and Testing
# ______________________________________________________________________________________
# Configuration script
##set(prefix ${CMAKE_INSTALL_PREFIX})
##set(exec_prefix "${prefix}")
##set(includedir "${prefix}/include")
##set(libdir "${prefix}/lib")
##install (TARGETS train DESTINATION bin)

# Testing session
#_____________________________________________________
#enable_testing()
#set (TEST_RUNCARD "${PROJECT_SOURCE_DIR}/tests/test_runcard.yaml")
#add_test(NAME Test COMMAND train TEST_RUNCARD)