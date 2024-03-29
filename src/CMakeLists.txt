set(src_files
    AnalyticCostFunction.cc
    IterationCallBack.cc
    )

add_library (src OBJECT ${src_files})

option (SHARED "Build shared-libray instead of static-libray" ON)
if (SHARED)
  add_library (NTK SHARED $<TARGET_OBJECTS:src>)
else (SHARED)
  add_library (NTK STATIC $<TARGET_OBJECTS:src>)
endif (SHARED)

target_link_libraries (NTK ${EIGEN3_LDFLAGS} ${CERES_LIBRARIES} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} ${YAML_LDFLAGS})

# Install libs
install (TARGETS NTK DESTINATION lib)
install (DIRECTORY ${PROJECT_SOURCE_DIR}/inc/NTK DESTINATION include)