set(ISPC_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/cost_function.ispc"
)

set(ISPC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ispc_lib.o")
set(ISPC_OUTPUT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/ispc_lib.h")

add_custom_command(
    OUTPUT ${ISPC_OUTPUT}
    COMMAND ${ISPC_EXECUTABLE} ${ISPC_OPTIONS}
                 -o ${ISPC_OUTPUT}
                 -h ${ISPC_OUTPUT_HEADER}
                 ${ISPC_SRCS}
    DEPENDS ${ISPC_SRCS}
    COMMENT "Building ISPC components"
    )

add_custom_target(ispc_target DEPENDS ${ISPC_OUTPUT})
add_library(ispc_lib INTERFACE IMPORTED GLOBAL)
add_dependencies(ispc_lib ispc_target)

set_target_properties(ispc_lib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_BINARY_DIR}
    INTERFACE_LINK_LIBRARIES ${ISPC_OUTPUT}
    )

