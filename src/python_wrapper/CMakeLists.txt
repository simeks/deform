pybind11_add_module(_pydeform _pydeform.cpp)

target_include_directories(_pydeform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${pybind11_INCLUDE_DIR})
target_link_libraries(_pydeform PUBLIC stk yaml-cpp gco deform_lib)

set(DF_PYTHON_PATH "$ENV{PYTHONPATH}${SEP}${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "Python path for Sphinx")

