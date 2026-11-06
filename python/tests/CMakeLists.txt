execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pytest --version
    RESULT_VARIABLE PYTEST_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(PYTEST_result EQUAL 0)
    add_custom_target(test_python_neural_set
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME test_python_neural_set
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_property(TEST test_python_neural_set PROPERTY
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:pyariadne>"
    )
    set_property(TEST test_python_neural_set PROPERTY LABELS "python")
    add_dependencies(test_python_neural_set pyariadne)
    add_dependencies(tests pyariadne)
elseif(PYTHON_ENFORCE_TESTS)
    message(FATAL_ERROR "pytest unavailable for ${Python3_EXECUTABLE}")
else()
    message(WARNING "pytest unavailable; Python tests skipped.")
endif()
