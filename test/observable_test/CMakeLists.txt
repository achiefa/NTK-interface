add_executable(unit_obs_test observable_test.cpp)
target_link_libraries(unit_obs_test PRIVATE NTK Catch2::Catch2WithMain)

add_executable(obs_benchmark observable_benchmark.cpp)
target_link_libraries(obs_benchmark PRIVATE NTK benchmark::benchmark)

add_executable(secret plain.cpp)
target_link_libraries(secret PRIVATE NTK)

catch_discover_tests(unit_obs_test)
AddCoverage(unit_obs_test)