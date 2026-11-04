cmake_minimum_required(VERSION 3.15)

project(NeuroSymbolic VERSION 0.1.0)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(NOT TARGET ariadne-configuration)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    submodule update --init -- ss/cfg
                    RESULT_VARIABLE _result)
    if(_result)
        message(FATAL_ERROR "Cannot initialize configuration submodule")
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/ss/cfg/cmake/ProjectOptions.cmake")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    require_same_dependency_commit(Configuration CHAINS
        "configuration"
        "neural|configuration"
        "symbolic|configuration"
    )
    require_same_dependency_commit(Testing CHAINS
        "testing"
        "neural|testing"
        "symbolic|testing"
    )
endif()

setup_project()

include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(ss)

# The CMake identifier omits '-' to match generated Python C++ entry points.
add_library(neurosymbolic INTERFACE)
target_include_directories(neurosymbolic INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ariadne-neuro-symbolic>
)
target_link_libraries(neurosymbolic INTERFACE neural symbolic)

add_subdirectory(src)
target_link_libraries(NEUROSYMBOLIC_SRC PRIVATE neurosymbolic)

ariadne_register_public_headers(
    TARGET neurosymbolic
    DIRECTORIES "${PROJECT_SOURCE_DIR}/include/neuro_symbolic"
    FILES "${PROJECT_SOURCE_DIR}/include/ariadne-neuro-symbolic.hpp"
)

setup_project_library(
    TARGET ariadne-neuro-symbolic
    OBJECTS NEUROSYMBOLIC_SRC
    LINK_LIBRARIES neurosymbolic
)

setup_standalone_project_tests(
    TARGET ariadne-neuro-symbolic
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
        TARGET neurosymbolic
        DESTINATION include/ariadne-neuro-symbolic
    )
endif()
