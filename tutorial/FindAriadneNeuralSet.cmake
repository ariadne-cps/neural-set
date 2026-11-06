find_library(ARIADNE_NEURAL_SET_LIBRARY
    NAMES ariadne-neural-set)
find_path(ARIADNE_NEURAL_SET_INCLUDE_DIR
    ariadne-neural-set.hpp
    PATH_SUFFIXES ariadne-neural-set)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AriadneNeuralSet DEFAULT_MSG
    ARIADNE_NEURAL_SET_LIBRARY
    ARIADNE_NEURAL_SET_INCLUDE_DIR
)

if(AriadneNeuralSet_FOUND)
    get_filename_component(ARIADNE_NEURAL_SET_INCLUDE_PARENT_DIR
        "${ARIADNE_NEURAL_SET_INCLUDE_DIR}" DIRECTORY)
    if(NOT TARGET AriadneNeuralSet::ariadne-neural-set)
        add_library(AriadneNeuralSet::ariadne-neural-set UNKNOWN IMPORTED)
        set_target_properties(AriadneNeuralSet::ariadne-neural-set PROPERTIES
            IMPORTED_LOCATION "${ARIADNE_NEURAL_SET_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${ARIADNE_NEURAL_SET_INCLUDE_PARENT_DIR};${ARIADNE_NEURAL_SET_INCLUDE_DIR}"
        )
    endif()
endif()
