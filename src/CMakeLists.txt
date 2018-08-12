set(DEFORM_LIB_SRCS
    "deform_lib/filters/gaussian_filter.cpp"
    "deform_lib/filters/gaussian_filter.h"
    "deform_lib/filters/resample.cpp"
    "deform_lib/filters/resample.h"

    "deform_lib/graph_cut/graph_cut.h"
    "deform_lib/graph_cut/graph_cut.inl"

    "deform_lib/graph_cut/graph_cut.h"
    "deform_lib/graph_cut/graph_cut.inl"

    "deform_lib/registration/block_change_flags.cpp"
    "deform_lib/registration/block_change_flags.h"
    "deform_lib/registration/blocked_graph_cut_optimizer.h"
    "deform_lib/registration/blocked_graph_cut_optimizer.inl"
    "deform_lib/registration/landmarks.cpp"
    "deform_lib/registration/landmarks.h"
    "deform_lib/registration/registration_engine.cpp"
    "deform_lib/registration/registration_engine.h"
    "deform_lib/registration/settings.cpp"
    "deform_lib/registration/settings.h"
    "deform_lib/registration/transform.cpp"
    "deform_lib/registration/transform.h"
    "deform_lib/registration/volume_pyramid.cpp"
    "deform_lib/registration/volume_pyramid.h"
    "deform_lib/registration/voxel_constraints.cpp"
    "deform_lib/registration/voxel_constraints.h"
    "deform_lib/registration/registration.cpp"
    "deform_lib/registration/registration.h"

    "deform_lib/arg_parser.cpp"
    "deform_lib/arg_parser.h"
    "deform_lib/config.h"
    "deform_lib/cost_function.h"
    "deform_lib/jacobian.cpp"
    "deform_lib/jacobian.h"
    "deform_lib/mutual_information.cpp"
    "deform_lib/mutual_information.h"
    "deform_lib/regularize.cpp"
)

set(DEFORM_SRCS
    "deform/jacobian.cpp"
    "deform/main.cpp"
    "deform/transform.cpp"
)

set(DEFORM_UTIL_SRCS
    "deform_util/main.cpp"
    "deform_util/cost.cpp"
)

if (DF_ENABLE_MICROPROFILE)
    set(DEFORM_LIB_SRCS 
        ${DEFORM_LIB_SRCS}
        "deform_lib/profiler/microprofile_html.h"
        "deform_lib/profiler/microprofile.cpp"
        "deform_lib/profiler/microprofile.h"
    )
endif()

if (DF_USE_CUDA)
    set(DEFORM_LIB_SRCS 
        ${DEFORM_LIB_SRCS}
        "deform_lib/filters/gpu/gaussian_filter.cu"
        "deform_lib/filters/gpu/gaussian_filter.h"
        "deform_lib/filters/gpu/resample.cu"
        "deform_lib/filters/gpu/resample.h"
    
        #"deform_lib/registration/gpu/cost_function.cu"
        #"deform_lib/registration/gpu/cost_function.h"
        "deform_lib/registration/gpu/transform.cu"
        "deform_lib/registration/gpu/transform.h"
        "deform_lib/registration/gpu_registration_engine.cpp"
        "deform_lib/registration/gpu_registration_engine.h"
        "deform_lib/registration/gpu_volume_pyramid.cpp"
        "deform_lib/registration/gpu_volume_pyramid.h"
    )
    CUDA_ADD_LIBRARY(deform_lib STATIC ${DEFORM_LIB_SRCS})
else ()
    add_library(deform_lib STATIC ${DEFORM_LIB_SRCS})
endif ()

if (DF_ENABLE_NVTOOLSEXT)
    target_link_libraries(deform_lib ${NvToolsExt_LIBRARIES})
    target_include_directories(deform_lib PUBLIC ${NvToolsExt_INCLUDE_DIRS})
endif()

add_library(deform_lib STATIC ${DEFORM_LIB_SRCS})
add_subdirectory(ispc)

target_include_directories(deform_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(deform_lib stk yaml-cpp gco ispc_lib)

if(WIN32)
  target_link_libraries(deform_lib wsock32 ws2_32)
endif()

add_executable(deform ${DEFORM_SRCS})
target_link_libraries(deform deform_lib)

add_executable(deform-util ${DEFORM_UTIL_SRCS})
target_link_libraries(deform-util deform_lib)

if(DF_BUILD_PYTHON_WRAPPER)
    pybind11_add_module(_pydeform "python_wrapper/_pydeform.cpp")
    target_include_directories(_pydeform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(_pydeform PUBLIC stk yaml-cpp gco deform_lib)

    set(DF_PYTHON_PATH "$ENV{PYTHONPATH}${SEP}${CMAKE_CURRENT_BINARY_DIR}" PARENT_SCOPE)
endif()
