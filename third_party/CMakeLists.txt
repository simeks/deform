
if (NOT YAML-CPP_FOUND)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Do not build yaml-cpp tests")
    add_subdirectory(yaml-cpp)
endif()

add_library(gco INTERFACE)
target_include_directories(gco INTERFACE gco-v3.0)

set(STK_USE_CUDA ${DF_USE_CUDA})
set(STK_WARNINGS_ARE_ERRORS ${DF_WARNINGS_ARE_ERRORS})
set(STK_BUILD_WITH_DEBUG_INFO ${DF_BUILD_WITH_DEBUG_INFO})
set(STK_ENABLE_FAST_MATH ${DF_ENABLE_FAST_MATH})

add_subdirectory(stk)

if(DF_BUILD_PYTHON_WRAPPER)
    add_subdirectory(pybind11)
endif()
