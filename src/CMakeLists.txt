set(src_files
    AnalyticCostFunction.cc
    IterationCallBack.cc
    )

add_library (src OBJECT ${src_files})
target_include_directories(src PUBLIC "${PROJECT_SOURCE_DIR}/inc/NTK")

option (SHARED "Build shared-libray instead of static-libray" ON)
if (SHARED)
  add_library (NTK SHARED $<TARGET_OBJECTS:src>)
else (SHARED)
  add_library (NTK STATIC $<TARGET_OBJECTS:src>)
endif (SHARED)

# Note
# I've added `src` into NTK so that the source files are
# included in the library `NTK`. Probably not the best solution...
target_link_libraries (NTK ${EIGEN3_LDFLAGS} ${CERES_LIBRARIES} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} ${YAML_LDFLAGS} src)

# Install libs
install (TARGETS NTK DESTINATION lib)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/inc/NTK DESTINATION include)