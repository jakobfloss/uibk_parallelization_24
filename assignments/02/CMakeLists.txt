cmake_minimum_required(VERSION 3.9)

project(assignments C CXX)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -fsanitize=undefined -Wall -Wextra -pedantic")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(mandelbrot_seq mandelbrot_seq.cpp)

add_executable(mandelbrot_mpi_reduce mandelbrot_mpi_reduce.cpp)
target_link_libraries(mandelbrot_mpi_reduce PUBLIC MPI::MPI_C)

add_executable(mandelbrot_mpi_gather mandelbrot_mpi_gather.cpp)
target_link_libraries(mandelbrot_mpi_gather PUBLIC MPI::MPI_C)
