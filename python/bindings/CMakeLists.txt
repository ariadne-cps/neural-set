include_directories(${Python3_INCLUDE_DIRS})
get_target_property(PYTHON_COMMON_SOURCE_DIR pyariadne-python-common SOURCE_DIR)

add_library(pyariadne-neurosymbolic-submodule-obj OBJECT neurosymbolic_submodule.cpp)
target_include_directories(pyariadne-neurosymbolic-submodule-obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PYTHON_COMMON_SOURCE_DIR}/include"
)
target_link_libraries(pyariadne-neurosymbolic-submodule-obj PRIVATE
    pyariadne-python-common
    pybind11::module
    neurosymbolic
)
set_target_properties(pyariadne-neurosymbolic-submodule-obj
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyariadne-neurosymbolic INTERFACE)
target_sources(pyariadne-neurosymbolic INTERFACE
    $<TARGET_OBJECTS:pyariadne-neurosymbolic-submodule-obj>
)
target_include_directories(pyariadne-neurosymbolic INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(pyariadne-neurosymbolic INTERFACE
    pyariadne-neural
    pyariadne-expression_set
)

ariadne_add_python_module(NAME neurosymbolic MODULE_IMPORT_NAME pyariadne_neurosymbolic MODULE_SOURCE neurosymbolic_module.cpp)
