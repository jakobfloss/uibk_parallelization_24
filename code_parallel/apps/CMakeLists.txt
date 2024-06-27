cmake_minimum_required(VERSION 3.10)


add_executable (run_full_code.serial
    main_full_code.cpp
    )

target_include_directories(run_full_code.serial
    PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(run_full_code.serial
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC solver
)



add_executable (run_singlestep
    main_test_singlesteps.cpp
    )

target_include_directories(run_singlestep
    PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(run_singlestep
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC solver
    PUBLIC IO
)



# Parallel version of code
add_executable (run_full_code.parallel
    main_full_code_parallel.cpp
    )

target_compile_definitions(run_full_code.parallel
    PRIVATE PARALLEL_VERSION
    PUBLIC OMPI_SKIP_MPICXX)

target_compile_options(run_full_code.parallel PRIVATE ${MPI_CXX_COMPILE_FLAGS})

target_include_directories(run_full_code.parallel
    PUBLIC ${CODE_INCLUDE_DIR}
    PUBLIC ${MPI_CXX_INCLUDE_PATH}
)

 
target_link_libraries(run_full_code.parallel
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC mpi_stuff
    PUBLIC solver_parallel
    PUBLIC ${MPI_CXX_LIBRARIES}
)
