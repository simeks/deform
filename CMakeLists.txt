cmake_minimum_required(VERSION 3.8)

option(DF_BUILD_TESTS "Build unit tests" ON)
option(DF_USE_CUDA "Enables CUDA support" OFF)
option(DF_WARNINGS_ARE_ERRORS "Warnings are treated as errors" OFF)
option(DF_BUILD_WITH_DEBUG_INFO "Includes debug info in release builds" OFF)
option(DF_ENABLE_FAST_MATH "Enables fast math" OFF)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting default build type: 'Release'")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

project(deform)

set(CMAKE_CXX_STANDARD 14)
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(EXTRA_FLAGS "-Wall")

    if (DF_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} -Werror")
    endif()

    if (DF_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "-g1")
    endif()

    if (DF_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(EXTRA_FLAGS "-Wall")

    if (DF_WARNINGS_ARE_ERRORS)
        set(EXTRA_FLAGS "${extra_flags} -Werror")
    endif()

    if (DF_BUILD_WITH_DEBUG_INFO)
        set(EXTRA_FLAGS_RELEASE "-g1")
    endif()

    if (DF_ENABLE_FAST_MATH)
        set(EXTRA_FLAGS "${extra_flags} -ffast-math")
    endif()
endif()

if (DF_USE_CUDA)
    add_definitions(-DDF_USE_CUDA)
    find_package(CUDA REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${EXTRA_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${EXTRA_FLAGS_RELEASE}")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${EXTRA_LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} ${EXTRA_LINK_FLAGS_RELEASE}")

add_subdirectory(third_party)
add_subdirectory(src)

if (DF_BUILD_TESTS)
    add_subdirectory(test)
endif()

