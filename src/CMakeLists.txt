add_library(NEUROSYMBOLIC_SRC OBJECT
    neuro_symbolic.cpp
    validated.cpp
    symbolic.cpp
)
target_include_directories(NEUROSYMBOLIC_SRC PRIVATE
    "${PROJECT_SOURCE_DIR}/include/neuro_symbolic"
)
