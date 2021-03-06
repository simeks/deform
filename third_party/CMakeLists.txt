set(STK_USE_CUDA ${DF_USE_CUDA} CACHE BOOL "")
set(STK_WARNINGS_ARE_ERRORS OFF CACHE BOOL "")
set(STK_BUILD_WITH_DEBUG_INFO ${DF_BUILD_WITH_DEBUG_INFO} CACHE BOOL "")
set(STK_ENABLE_FAST_MATH ${DF_ENABLE_FAST_MATH} CACHE BOOL "")
set(STK_ITK_BRIDGE ${DF_ITK_BRIDGE} CACHE BOOL "")
set(STK_STACK_TRACE ${DF_STACK_TRACE} CACHE BOOL "")
set(STK_BUILD_PYTHON_WRAPPER ${DF_BUILD_PYTHON_WRAPPER} CACHE BOOL "")
set(STK_BUILD_EXAMPLES OFF CACHE BOOL "")
set(STK_BUILD_TESTS OFF CACHE BOOL "")

add_subdirectory(stk)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(REPLACE "-Wextra" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Wextra" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    string(REPLACE "-pedantic" ""  CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-pedantic" ""  CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

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


if (DF_ENABLE_GCO)
    add_library(gco INTERFACE)
    target_include_directories(gco INTERFACE gco-v3.0)
endif()

if (DF_ENABLE_GRIDCUT)
    add_library(gridcut INTERFACE)
    target_include_directories(gridcut INTERFACE GridCut-1.3/include)
endif()
