pybind11_add_module(_pydeform _pydeform.cpp)

target_include_directories(_pydeform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${pybind11_INCLUDE_DIR})
target_link_libraries(_pydeform PUBLIC stk yaml-cpp::yaml-cpp gco deform_lib)

install(TARGETS _pydeform
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)