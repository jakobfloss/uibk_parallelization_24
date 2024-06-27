add_library(utilities
    matrix.cpp
)

target_include_directories(utilities
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)