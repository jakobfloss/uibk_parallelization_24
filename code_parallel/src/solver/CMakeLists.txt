add_library(solver
    singlestep.cpp
    singlestep_flux_based.cpp
    Riemann_solvers.cpp
    Riemann_solver_HLL.cpp
    finite_volume_solver.cpp
    reconstruction.cpp
    limiter.cpp
    time_integrator.cpp
)

target_include_directories(solver
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)


target_link_libraries(solver
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC IO
)


# To be COPIED by students
add_library(solver_parallel
    singlestep.cpp
    singlestep_flux_based.cpp
    Riemann_solvers.cpp
    Riemann_solver_HLL.cpp
    finite_volume_solver.cpp
    reconstruction.cpp
    limiter.cpp
    time_integrator.cpp
)

target_include_directories(solver_parallel
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)


target_link_libraries(solver_parallel
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC IO
    PUBLIC IO_parallel
)

target_compile_definitions(solver_parallel
  PRIVATE PARALLEL_VERSION
  PUBLIC OMPI_SKIP_MPICXX
  )

target_compile_options(solver_parallel PRIVATE ${MPI_CXX_COMPILE_FLAGS})
