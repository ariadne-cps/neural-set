include(CTest)

set(UNIT_TESTS
    test_neural_set
    test_validated
    test_taylor
    test_umbrella
)
foreach(TEST IN LISTS UNIT_TESTS)
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} PRIVATE
        ariadne-neural-set neural_set ariadne-testing)
    add_test(NAME ${TEST} COMMAND $<TARGET_FILE:${TEST}> --concise)
    set_property(TEST ${TEST} PROPERTY LABELS "neural_set")
    if(WIN32)
        set_tests_properties(${TEST} PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:ariadne-neural-set>"
        )
    endif()
endforeach()
add_custom_target(tests)
add_dependencies(tests ${UNIT_TESTS})
