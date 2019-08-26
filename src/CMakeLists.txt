add_subdirectory(deform_lib)

if(DF_BUILD_EXECUTABLE)
    add_subdirectory(deform)
endif()

if(DF_BUILD_PYTHON_WRAPPER)
    add_subdirectory(python_wrapper)
endif()
