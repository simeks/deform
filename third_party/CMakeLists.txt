
if (NOT YAML-CPP_FOUND)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Do not build yaml-cpp tests")
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Do not build yaml-cpp tools")
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Do not build yaml-cpp contrib")
    add_subdirectory(yaml-cpp)

    if (WIN32)
        # /wd4100 : "'type': unreferenced formal parameter"
        # /wd4127 : "conditional expression is constant"
        # /wd4244 : "conversion from 'unsigned int' to 'unsigned char'"
        # /wd4456 : "declaration of 'token' hides previous local declaration"
        # /wd4702 : "unreachable code"
        set_target_properties(yaml-cpp PROPERTIES COMPILE_FLAGS 
            "/wd4100 /wd4127 /wd4244 /wd4456 /wd4702"
            )
    endif()
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
