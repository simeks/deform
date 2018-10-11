set(DEFORM_UTIL_SRCS
    "main.cpp"
    "cost.cpp"
)

add_executable(deform-util ${DEFORM_UTIL_SRCS})
target_link_libraries(deform-util deform_lib)
