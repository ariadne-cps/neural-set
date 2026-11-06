add_library(NEURAL_SET_SRC OBJECT
    neural_set.cpp
    validated.cpp
    taylor.cpp
)
target_include_directories(NEURAL_SET_SRC PRIVATE
    "${PROJECT_SOURCE_DIR}/include/neural-set"
)
