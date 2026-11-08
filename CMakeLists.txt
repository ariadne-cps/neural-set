cmake_minimum_required(VERSION 3.15)

project(NeuralSet VERSION 0.1.0)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(NOT TARGET ariadne-build)
    find_package(Git REQUIRED)
    # Keep the initial Build checkout output in English.
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "LC_ALL=C"
                "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                submodule update --init -- ss/bui
        RESULT_VARIABLE _result)
    if(_result)
        message(FATAL_ERROR "Cannot initialize build submodule")
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/ss/bui/cmake/ProjectOptions.cmake")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Pinned child commits guarantee their internal consistency.
    require_same_dependency_commit(Build CHAINS
        "build" "ann|build" "set|build")
endif()

setup_project()

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(ss)

# The CMake identifier omits '-' to match generated Python C++ entry points.
add_library(neural_set INTERFACE)
target_include_directories(neural_set INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ariadne-neural-set>
)
target_link_libraries(neural_set INTERFACE ann set)

add_subdirectory(src)
target_link_libraries(NEURAL_SET_SRC PRIVATE neural_set)

ariadne_register_public_headers(
    TARGET neural_set
    DIRECTORIES "${PROJECT_SOURCE_DIR}/include/neural-set"
    FILES "${PROJECT_SOURCE_DIR}/include/ariadne-neural-set.hpp"
)

setup_project_library(
    TARGET ariadne-neural-set
    OBJECTS NEURAL_SET_SRC
    LINK_LIBRARIES neural_set
)

setup_standalone_project(
    TARGET ariadne-neural-set
    TEST_DIRECTORY tests
    TEST_TARGET tests
    SOURCES
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
)

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    ariadne_install_dependency_bundle(
        TARGET neural_set
        DESTINATION include/ariadne-neural-set
    )
endif()
