add_subdirectory(deform_lib)
add_subdirectory(deform)
add_subdirectory(deform_util)

if(DF_BUILD_PYTHON_WRAPPER)
    add_subdirectory(python_wrapper)
endif()
