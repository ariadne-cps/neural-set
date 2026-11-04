if(NOT TARGET neural)
    ariadne_add_submodule(neural)
endif()

if(NOT TARGET expression_set)
    ariadne_add_submodule(expression-set)
endif()

# Testing remains a test-only dependency.
ariadne_setup_testing()
