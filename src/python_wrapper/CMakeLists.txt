add_library(_pydeform MODULE "_pydeform.cpp")

set_target_properties(_pydeform PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    )

target_include_directories(_pydeform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIR} ${pybind11_INCLUDE_DIR})
target_link_libraries(_pydeform PUBLIC stk yaml-cpp gco deform_lib ${PYTHON_LIBRARIES})

set(DF_PYTHON_PATH "$ENV{PYTHONPATH}${SEP}${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
