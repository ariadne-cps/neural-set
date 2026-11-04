find_library(ARIADNE_NEUROSYMBOLIC_LIBRARY
    NAMES ariadne-neurosymbolic)
find_path(ARIADNE_NEUROSYMBOLIC_INCLUDE_DIR
    ariadne-neurosymbolic.hpp
    PATH_SUFFIXES ariadne-neurosymbolic)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AriadneNeurosymbolic DEFAULT_MSG
    ARIADNE_NEUROSYMBOLIC_LIBRARY
    ARIADNE_NEUROSYMBOLIC_INCLUDE_DIR
)

if(AriadneNeurosymbolic_FOUND)
    get_filename_component(ARIADNE_NEUROSYMBOLIC_INCLUDE_PARENT_DIR
        "${ARIADNE_NEUROSYMBOLIC_INCLUDE_DIR}" DIRECTORY)
    if(NOT TARGET AriadneNeurosymbolic::ariadne-neurosymbolic)
        add_library(AriadneNeurosymbolic::ariadne-neurosymbolic UNKNOWN IMPORTED)
        set_target_properties(AriadneNeurosymbolic::ariadne-neurosymbolic PROPERTIES
            IMPORTED_LOCATION "${ARIADNE_NEUROSYMBOLIC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${ARIADNE_NEUROSYMBOLIC_INCLUDE_PARENT_DIR};${ARIADNE_NEUROSYMBOLIC_INCLUDE_DIR}"
        )
    endif()
endif()
