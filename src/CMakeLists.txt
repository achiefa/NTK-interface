set(src_files
    AnalyticCostFunction.cc
    IterationCallBack.cc
    NumericalDerivative.cc
    Observable.cc
    )

add_library (src OBJECT ${src_files})

option (SHARED "Build shared-libray instead of static-libray" ON)
if (SHARED)
  add_library (NTK SHARED $<TARGET_OBJECTS:src>)
else (SHARED)
  add_library (NTK STATIC $<TARGET_OBJECTS:src>)
endif (SHARED)

add_library (NTK STATIC $<TARGET_OBJECTS:src>)

target_link_libraries (NTK ${EIGEN3_LDFLAGS} ${CERES_LIBRARIES} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} ${YAML_LDFLAGS})

if (CMAKE_BUILD_TYPE STREQUAL Debug)
  target_compile_options(NTK PRIVATE --coverage)
  target_link_options(NTK PUBLIC --coverage)
  add_custom_command(TARGET NTK PRE_BUILD
  COMMAND find ${CMAKE_BINARY_DIR} -type f -name '*.gcda' -exec rm {} +)
endif()



# Install libs
install (TARGETS NTK DESTINATION lib)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/inc/NTK DESTINATION include)

