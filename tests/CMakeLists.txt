include(CTest)

set(UNIT_TESTS
    test_neurosymbolic
    test_validated
    test_taylor
    test_symbolic
    test_umbrella
)
foreach(TEST IN LISTS UNIT_TESTS)
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} PRIVATE
        ariadne-neurosymbolic neurosymbolic ariadne-testing)
    add_test(NAME ${TEST} COMMAND $<TARGET_FILE:${TEST}> --concise)
    set_property(TEST ${TEST} PROPERTY LABELS "neurosymbolic")
    if(WIN32)
        set_tests_properties(${TEST} PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:ariadne-neurosymbolic>"
        )
    endif()
endforeach()
add_custom_target(tests)
add_dependencies(tests ${UNIT_TESTS})

# Compile and execute the public C++ tutorial as part of the regression.
# This catches examples that build only with private source-tree headers.
add_executable(test_tutorial_cpp ../tutorial/tutorial_neurosymbolic.cpp)
target_link_libraries(test_tutorial_cpp PRIVATE
    ariadne-neurosymbolic neurosymbolic ariadne-testing)
add_test(NAME test_tutorial_cpp COMMAND $<TARGET_FILE:test_tutorial_cpp>)
set_property(TEST test_tutorial_cpp PROPERTY LABELS "tutorial")
if(WIN32)
    set_tests_properties(test_tutorial_cpp PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:ariadne-neurosymbolic>"
    )
endif()
add_dependencies(tests test_tutorial_cpp)
