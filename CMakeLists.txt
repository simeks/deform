cmake_minimum_required(VERSION 3.8)

project(deform)

# path separator
if(WIN32)
  set(SEP "\\;")
else()
  set(SEP ":")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Set location of outputs

if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif ()
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

set(DF_VERSION_FILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/deform_lib/version.h.in")
message(${DF_VERSION_FILE_IN})
set(DF_VERSION_FILE_OUT "src/deform_lib/version.h")
message(${DF_VERSION_FILE_OUT})

include(ProjectVersion)

option(DF_BUILD_TESTS "Build unit tests" OFF)
option(DF_BUILD_DOCS "Build Sphinx docs" OFF)
option(DF_BUILD_EXECUTABLE "Build registration executable" ON)
option(DF_BUILD_UTILS "Build utils executable" ON)
option(DF_BUILD_PYTHON_WRAPPER "Build Python wrapper" OFF)
option(DF_USE_CUDA "Enable CUDA support" OFF)
option(DF_USE_ISPC "Enable ISPC support" OFF)
option(DF_WARNINGS_ARE_ERRORS "Warnings are treated as errors" OFF)
option(DF_BUILD_WITH_DEBUG_INFO "Include debug info in release builds" OFF)
option(DF_ENABLE_FAST_MATH "Enable fast math" OFF)
option(DF_ITK_BRIDGE "Add support to interoperate with ITK" OFF)
option(DF_STACK_TRACE "Print a stack trace on errors" OFF)
option(DF_SIGNAL_HANDLER "Enable signal handling (unsafe, for debug only)" OFF)
option(DF_NON_PORTABLE "Enable non-portable optimisations" ON)

option(DF_ENABLE_MICROPROFILE "Enable microprofile profiler" OFF)
option(DF_ENABLE_NVTOOLSEXT "Enable nvtoolsext profiler" OFF)

if (DF_ENABLE_MICROPROFILE AND DF_ENABLE_NVTOOLSEXT)
    message(FATAL_ERROR "Can't enable multiple profilers at the same time")
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(DF_SIGNAL_HANDLER)
    set(DF_STACK_TRACE ON)
    add_definitions(-DDF_SIGNAL_HANDLER)
endif()

if(DF_STACK_TRACE)
    set(DF_BUILD_WITH_DEBUG_INFO ON)
    if(UNIX)
        find_package(Backward)
    endif()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (MSVC)
    add_definitions( -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_DEPRECATE )
    set(EXTRA_FLAGS "/FS")

    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "/W[1-3]" "/W4" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

    if (DF_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} /WX")
    endif()

    if (DF_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "/Zi")
        set(EXTRA_LINK_FLAGS_RELEASE "/DEBUG")
    endif()

    if (DF_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} /fp:fast")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(EXTRA_FLAGS "-Wall -Wextra -pedantic")

    if(DF_NON_PORTABLE)
        set(EXTRA_FLAGS "${extra_flags} -march=native")
    endif()

    if (DF_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} -Werror")
    endif()

    if (DF_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "-g1 -rdynamic")
    endif()

    if (DF_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
endif()

if (DF_USE_CUDA)
    add_definitions(-DDF_USE_CUDA)
    add_definitions(-DSTK_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)

    if (DF_ENABLE_FAST_MATH)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    endif()

    if (DF_ENABLE_NVTOOLSEXT)
        find_package(NvToolsExt REQUIRED)
        add_definitions(-DDF_ENABLE_NVTOOLSEXT)
    endif()
endif()

if (DF_ENABLE_MICROPROFILE)
    add_definitions(-DDF_ENABLE_MICROPROFILE)
endif()

if (DF_USE_ISPC)
    add_definitions(-DDF_USE_ISPC)
    find_package(ISPC REQUIRED)

    if(NOT WIN32)
        set(ISPC_OPTIONS "--pic")
    endif()

    if (DF_ENABLE_FAST_MATH)
        set(ISPC_OPTIONS ${ISPC_OPTIONS} " --opt=fast-math")
    endif()
    if (DF_BUILD_WITH_DEBUG_INFO)
        set(ISPC_OPTIONS ${ISPC_OPTIONS} " -g")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_LINK_FLAGS_RELEASE}")

include_directories(src)

if(NOT WIN32)
    find_package(YAML-CPP QUIET)
endif()

add_subdirectory(third_party)
add_subdirectory(src)

if (DF_BUILD_DOCS)
    set(SETUP_PY "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
    add_subdirectory(sphinx)
endif()

if (DF_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
