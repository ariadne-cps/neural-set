find_library(ARIADNE_NEURO_SYMBOLIC_LIBRARY
    NAMES ariadne-neuro-symbolic)
find_path(ARIADNE_NEURO_SYMBOLIC_INCLUDE_DIR
    ariadne-neuro-symbolic.hpp
    PATH_SUFFIXES ariadne-neuro-symbolic)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AriadneNeuroSymbolic DEFAULT_MSG
    ARIADNE_NEURO_SYMBOLIC_LIBRARY
    ARIADNE_NEURO_SYMBOLIC_INCLUDE_DIR
)

if(AriadneNeuroSymbolic_FOUND)
    get_filename_component(ARIADNE_NEURO_SYMBOLIC_INCLUDE_PARENT_DIR
        "${ARIADNE_NEURO_SYMBOLIC_INCLUDE_DIR}" DIRECTORY)
    if(NOT TARGET AriadneNeuroSymbolic::ariadne-neuro-symbolic)
        add_library(AriadneNeuroSymbolic::ariadne-neuro-symbolic UNKNOWN IMPORTED)
        set_target_properties(AriadneNeuroSymbolic::ariadne-neuro-symbolic PROPERTIES
            IMPORTED_LOCATION "${ARIADNE_NEURO_SYMBOLIC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${ARIADNE_NEURO_SYMBOLIC_INCLUDE_PARENT_DIR};${ARIADNE_NEURO_SYMBOLIC_INCLUDE_DIR}"
        )
    endif()
endif()
