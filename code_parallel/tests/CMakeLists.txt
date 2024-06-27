add_executable(test_limiter test_limiter.cpp)
target_include_directories(test_limiter PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_limiter PRIVATE solver PRIVATE gtest PRIVATE gtest_main  PRIVATE pthread)
add_test(NAME test_limiter COMMAND test_limiter)

add_executable(test_physics test_physics.cpp)
target_include_directories(test_physics PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_physics PRIVATE sim_setup PRIVATE solver PRIVATE utilities PRIVATE gtest PRIVATE gtest_main  PRIVATE pthread)
add_test(NAME test_physics COMMAND test_physics)

add_executable(test_Riemann_solvers tests_Riemann_solvers.cpp)
target_include_directories(test_Riemann_solvers PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_Riemann_solvers PRIVATE sim_setup PRIVATE solver PRIVATE utilities PRIVATE gtest PRIVATE gtest_main  PRIVATE pthread)
add_test(NAME test_Riemann_solvers COMMAND test_Riemann_solvers)

add_executable(test_time_integrator test_time_integrator.cpp)
target_include_directories(test_time_integrator PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_time_integrator PRIVATE sim_setup PRIVATE solver PRIVATE utilities PRIVATE gtest PRIVATE gtest_main  PRIVATE pthread)
add_test(NAME test_time_integrator COMMAND test_time_integrator)
