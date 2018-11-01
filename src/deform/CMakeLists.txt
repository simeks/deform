set(DEFORM_SRCS
    "command.h"
    "divergence.cpp"
    "jacobian.cpp"
    "main.cpp"
    "registration.cpp"
    "regularize.cpp"
    "transform.cpp"
)

add_executable(deform ${DEFORM_SRCS} ${BACKWARD_ENABLE})

if(Backward_FOUND)
    add_backward(deform)
    add_definitions(-DDF_USE_BACKWARD)
endif()

target_link_libraries(deform deform_lib)
target_include_directories(deform PRIVATE "${CMAKE_BINARY_DIR}/src")

