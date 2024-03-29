add_executable(train train.cc)
target_link_libraries(train NTK)
target_compile_definitions (train PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang>:COMPILING_CXX_WITH_CLANG>"
    "$<$<COMPILE_LANG_AND_ID:CXX,Intel>:COMPILING_CXX_WITH_INTEL>"
)

target_compile_options (train INTERFACE 
    "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>"
    )