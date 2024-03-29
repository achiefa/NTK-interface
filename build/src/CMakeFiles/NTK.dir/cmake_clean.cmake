file(REMOVE_RECURSE
  "libNTK.dylib"
  "libNTK.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/NTK.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
