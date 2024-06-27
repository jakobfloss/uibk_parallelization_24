add_library(IO 
    data_storage.cpp
)

target_include_directories(IO
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(IO
    PUBLIC SerialHdf5
)



add_library(IO_parallel 
    data_storage_parallel.cpp
)

target_compile_definitions(IO_parallel
  PRIVATE PARALLEL_VERSION
  PUBLIC OMPI_SKIP_MPICXX
  )

target_compile_options(IO_parallel PRIVATE ${MPI_CXX_COMPILE_FLAGS})


target_include_directories(IO_parallel
    PUBLIC ${HDF5_INCLUDE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${MPI_CXX_INCLUDE_PATH}
)

target_link_libraries(IO_parallel
    PUBLIC ${HDF5_LIBRARIES}
)
