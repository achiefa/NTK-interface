add_executable(second_derivative second_derivative.cc)
target_link_libraries(second_derivative NTK)


# Testing session
#_____________________________________________________
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
  GIT_TAG         v3.6.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
include(Coverage)
add_subdirectory(observable_test)
