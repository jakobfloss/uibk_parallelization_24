add_library(sim_setup
    grid1D.cpp
    grid.cpp
    fluid.cpp
    physics.cpp
)

target_include_directories(sim_setup
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)


add_library(mpi_stuff
    mpi_handler.cpp
)

target_include_directories(mpi_stuff
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${MPI_CXX_INCLUDE_PATH}
)

target_compile_definitions(mpi_stuff
    PUBLIC OMPI_SKIP_MPICXX
)

target_compile_options(mpi_stuff PRIVATE ${MPI_CXX_COMPILE_FLAGS})

