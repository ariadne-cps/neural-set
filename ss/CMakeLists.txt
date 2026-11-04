if(NOT TARGET neural)
    ariadne_add_submodule(neural)
endif()

if(NOT TARGET symbolic)
    ariadne_add_submodule(symbolic)
endif()

# Testing remains a test-only dependency.
ariadne_setup_testing()
