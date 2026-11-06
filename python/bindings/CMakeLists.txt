include_directories(${Python3_INCLUDE_DIRS})
get_target_property(PYTHON_COMMON_SOURCE_DIR pyariadne-python-common SOURCE_DIR)

add_library(pyariadne-neural_set-submodule-obj OBJECT neural_set_submodule.cpp)
target_include_directories(pyariadne-neural_set-submodule-obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PYTHON_COMMON_SOURCE_DIR}/include"
)
target_link_libraries(pyariadne-neural_set-submodule-obj PRIVATE
    pyariadne-python-common
    pybind11::module
    neural_set
)
set_target_properties(pyariadne-neural_set-submodule-obj
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyariadne-neural_set INTERFACE)
target_sources(pyariadne-neural_set INTERFACE
    $<TARGET_OBJECTS:pyariadne-neural_set-submodule-obj>
)
target_include_directories(pyariadne-neural_set INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(pyariadne-neural_set INTERFACE
    pyariadne-ann
    pyariadne-set
)

ariadne_add_python_module(NAME neural_set MODULE_IMPORT_NAME pyariadne_neural_set MODULE_SOURCE neural_set_module.cpp)
