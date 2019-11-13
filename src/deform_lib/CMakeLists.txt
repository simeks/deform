set(DEFORM_LIB_SRCS
    "filters/gaussian_filter.cpp"
    "filters/gaussian_filter.h"
    "filters/resample.cpp"
    "filters/resample.h"

    "registration/cost_functions/cost_function.h"
    "registration/cost_functions/cross_correlation_cube.h"
    "registration/cost_functions/cross_correlation_sphere.h"
    "registration/cost_functions/landmarks.h"
    "registration/cost_functions/mutual_information.cpp"
    "registration/cost_functions/mutual_information.h"
    "registration/cost_functions/regularizer.h"
    "registration/cost_functions/soft_constraints.h"
    "registration/cost_functions/squared_distance.h"
    "registration/cost_functions/sub_function.h"
    "registration/cost_functions/unary_function.h"

    "registration/affine_transform.cpp"
    "registration/affine_transform.h"
    "registration/block_change_flags.cpp"
    "registration/block_change_flags.h"
    "registration/blocked_graph_cut_optimizer.h"
    "registration/blocked_graph_cut_optimizer.inl"
    "registration/displacement_field.cpp"
    "registration/displacement_field.h"
    "registration/landmarks.cpp"
    "registration/landmarks.h"
    "registration/registration_engine.cpp"
    "registration/registration_engine.h"
    "registration/settings.cpp"
    "registration/settings.h"
    "registration/transform.cpp"
    "registration/transform.h"
    "registration/volume_pyramid.cpp"
    "registration/volume_pyramid.h"
    "registration/voxel_constraints.cpp"
    "registration/voxel_constraints.h"
    "registration/registration.cpp"
    "registration/registration.h"

    "arg_parser.cpp"
    "arg_parser.h"
    "config.h"
    "jacobian.cpp"
    "jacobian.h"
    "regularize.cpp"
    "version.cpp"
    "version.h"
)

if (DF_ENABLE_GCO)
    set(DEFORM_LIB_SRCS
        ${DEFORM_LIB_SRCS}
        "solver/gco_solver.h"
        "solver/gco_solver.inl"
    )
endif ()

if (DF_ENABLE_GRIDCUT)
    set(DEFORM_LIB_SRCS
        ${DEFORM_LIB_SRCS}
        "solver/gridcut_solver.h"
    )
endif()

if (DF_USE_CUDA)
    set(DEFORM_LIB_SRCS
        ${DEFORM_LIB_SRCS}
        "filters/gpu/gaussian_filter.cu"
        "filters/gpu/gaussian_filter.h"
        "filters/gpu/resample.cu"
        "filters/gpu/resample.h"

        "registration/gpu/cost_functions/binary_function.cu"
        "registration/gpu/cost_functions/binary_function.h"
        "registration/gpu/cost_functions/cost_function_kernel.h"
        "registration/gpu/cost_functions/cross_correlation.cu"
        "registration/gpu/cost_functions/cross_correlation.h"
        "registration/gpu/cost_functions/landmarks.cu"
        "registration/gpu/cost_functions/landmarks.h"
        "registration/gpu/cost_functions/squared_distance.cu"
        "registration/gpu/cost_functions/squared_distance.h"
        "registration/gpu/cost_functions/cost_function.h"
        "registration/gpu/cost_functions/unary_function.h"

        "registration/gpu/gpu_displacement_field.cpp"
        "registration/gpu/gpu_displacement_field.cu"
        "registration/gpu/gpu_displacement_field.h"
        "registration/gpu/transform.cu"
        "registration/gpu/transform.h"
        "registration/gpu_registration_engine.cpp"
        "registration/gpu_registration_engine.h"
        "registration/gpu_volume_pyramid.cpp"
        "registration/gpu_volume_pyramid.h"
        "registration/hybrid_graph_cut_optimizer.inl"
        "registration/hybrid_graph_cut_optimizer.cu"
        "registration/hybrid_graph_cut_optimizer.h"
    )
endif ()

if (DF_ENABLE_MICROPROFILE)
    set(DEFORM_LIB_SRCS
        ${DEFORM_LIB_SRCS}
        "profiler/microprofile_html.h"
        "profiler/microprofile.cpp"
        "profiler/microprofile.h"
    )
endif()


add_library(deform_lib STATIC ${DEFORM_LIB_SRCS})

add_dependencies(deform_lib FetchGitVersion)

target_include_directories(deform_lib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} "${CMAKE_BINARY_DIR}/src")

target_link_libraries(deform_lib stk yaml-cpp)

if (DF_ENABLE_GCO)
    target_link_libraries(deform_lib gco)
endif()

if (DF_ENABLE_GRIDCUT)
    target_link_libraries(deform_lib gridcut)
endif()

if (DF_ENABLE_NVTOOLSEXT)
    target_link_libraries(deform_lib ${NvToolsExt_LIBRARIES})
    target_include_directories(deform_lib PUBLIC ${NvToolsExt_INCLUDE_DIRS})
endif()

if(DF_USE_ISPC)
    add_subdirectory(ispc)
    target_link_libraries(deform_lib ispc_lib)
endif()

if(WIN32)
  target_link_libraries(deform_lib wsock32 ws2_32)
endif()

