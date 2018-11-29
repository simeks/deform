set(DEFORM_TEST_SRCS 
    "test_arg_parser.cpp"
    "test_gpu.cpp"
    "test_gpu_cost.cpp"
    "test_main.cpp"
    "test_settings.cpp"
    "test_landmarks.cpp"
)

add_executable(deform_test ${DEFORM_TEST_SRCS})
add_test(deform_test ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/deform_test)

target_link_libraries(deform_test Catch deform_lib)

