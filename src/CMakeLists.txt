set(DEFORM_LIB_SRCS
    "deform_lib/cost_functions/cost_function.h"
    "deform_lib/cost_functions/cross_correlation_cube.h"
    "deform_lib/cost_functions/cross_correlation_sphere.h"
    "deform_lib/cost_functions/landmarks.h"
    "deform_lib/cost_functions/mutual_information.cpp"
    "deform_lib/cost_functions/mutual_information.h"
    "deform_lib/cost_functions/regularizer.h"
    "deform_lib/cost_functions/soft_constraints.h"
    "deform_lib/cost_functions/squared_distance.h"
    "deform_lib/cost_functions/sub_function.h"
    "deform_lib/cost_functions/unary_function.h"

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
    "deform_lib/jacobian.cpp"
    "deform_lib/jacobian.h"
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
    
        "deform_lib/registration/gpu/cost_function.cu"
        "deform_lib/registration/gpu/cost_function.h"
        "deform_lib/registration/gpu/transform.cu"
        "deform_lib/registration/gpu/transform.h"
        "deform_lib/registration/gpu_registration_engine.cpp"
        "deform_lib/registration/gpu_registration_engine.h"
        "deform_lib/registration/gpu_volume_pyramid.cpp"
        "deform_lib/registration/gpu_volume_pyramid.h"
        "deform_lib/registration/hybrid_graph_cut_optimizer.cpp"
        "deform_lib/registration/hybrid_graph_cut_optimizer.cu"
        "deform_lib/registration/hybrid_graph_cut_optimizer.h"
    )
endif ()

add_library(deform_lib STATIC ${DEFORM_LIB_SRCS})

if (DF_ENABLE_NVTOOLSEXT)
    target_link_libraries(deform_lib ${NvToolsExt_LIBRARIES})
    target_include_directories(deform_lib PUBLIC ${NvToolsExt_INCLUDE_DIRS})
endif()

target_include_directories(deform_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
target_link_libraries(deform_lib stk yaml-cpp gco)

if(DF_USE_ISPC)
    add_subdirectory(ispc)
    target_link_libraries(deform_lib ispc_lib)
endif()

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
