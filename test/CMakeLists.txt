set(DEFORM_TEST_SRCS 
    "test_main.cpp"
    "test_settings.cpp"
)

add_executable(deform_test ${DEFORM_TEST_SRCS})
add_test(deform_test deform_test)

target_link_libraries(deform_test Catch deform_lib)

