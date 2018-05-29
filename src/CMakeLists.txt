set(DEFORM_LIB_SRCS 
    "deform_lib/filters/gaussian_filter.cpp"
    "deform_lib/filters/gaussian_filter.h"
    "deform_lib/filters/resample.cpp"
    "deform_lib/filters/resample.h"

    "deform_lib/graph_cut/graph_cut.h"
    "deform_lib/graph_cut/graph_cut.inl"

    "deform_lib/graph_cut/graph_cut.h"
    "deform_lib/graph_cut/graph_cut.inl"
    
    "deform_lib/platform/file_path.cpp"
    "deform_lib/platform/file_path.h"
    "deform_lib/platform/gcc.h"
    #"deform_lib/platform/timer_posix.cpp"
    "deform_lib/platform/timer_win.cpp"
    "deform_lib/platform/timer.h"
    "deform_lib/platform/windows_wrapper.h"

    "deform_lib/registration/block_change_flags.cpp"
    "deform_lib/registration/block_change_flags.h"
    "deform_lib/registration/blocked_graph_cut_optimizer.h"
    "deform_lib/registration/blocked_graph_cut_optimizer.inl"
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

    "deform_lib/args.cpp"
    "deform_lib/args.h"
    "deform_lib/benchmark.cpp"
    "deform_lib/config.h"
    "deform_lib/cost_function.h"
    "deform_lib/jacobian.cpp"
    "deform_lib/jacobian.h"
    "deform_lib/regularize.cpp"
)
set(DEFORM_SRCS 
    "deform/jacobian.cpp"
    "deform/main.cpp"
    "deform/transform.cpp"
)

add_library(deform_lib STATIC ${DEFORM_LIB_SRCS})
target_include_directories(deform_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(deform_lib stk json gco)

add_executable(deform ${DEFORM_SRCS})
target_link_libraries(deform deform_lib)