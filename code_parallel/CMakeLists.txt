cmake_minimum_required(VERSION 3.10)

# set the project name
project(FluidSimulator
  LANGUAGES C CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# specify the C standard
set(C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# set standard flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -g2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mtune=native -DNDEBUG")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED)
MESSAGE(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIR}")

# Need MPI for parallel version - to be used in default setup
MESSAGE(STATUS "Searching for MPI package")
find_package(MPI REQUIRED)
MESSAGE(STATUS " Found include path at ${MPI_CXX_INCLUDE_PATH}")


# Make serial form of the library
add_library(SerialHdf5 INTERFACE)
target_link_libraries(SerialHdf5
  INTERFACE ${HDF5_LIBRARIES}
  )

if(${HDF5_IS_PARALLEL})
  
  find_package(MPI)
  MESSAGE(" Loaded MPI for parallel HDF5 ")
  MESSAGE(STATUS " Found include path at ${MPI_CXX_INCLUDE_PATH}")

  target_include_directories(SerialHdf5
    INTERFACE ${HDF5_INCLUDE_DIR}
    INTERFACE ${MPI_CXX_INCLUDE_PATH}
    )
  target_compile_definitions(SerialHdf5 INTERFACE OMPI_SKIP_MPICXX)

else()

  target_include_directories(SerialHdf5
    INTERFACE ${HDF5_INCLUDE_DIR}
    )
  
endif()


set(CODE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(apps)



if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
		      

  enable_testing()
  #find_package(GTest CONFIG REQUIRED)  # use googletest
  add_subdirectory(tests)
endif()
