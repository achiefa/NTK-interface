# CMake generated Testfile for 
# Source directory: /Users/s2569857/Codes/NTK-interface
# Build directory: /Users/s2569857/Codes/NTK-interface/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(Test "/Users/s2569857/Codes/NTK-interface/build/run/train" "TEST_RUNCARD")
set_tests_properties(Test PROPERTIES  _BACKTRACE_TRIPLES "/Users/s2569857/Codes/NTK-interface/CMakeLists.txt;76;add_test;/Users/s2569857/Codes/NTK-interface/CMakeLists.txt;0;")
subdirs("src")
subdirs("run")
