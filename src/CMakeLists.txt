set(DEFORM_SRCS 
    "deform/filters/gaussian_filter.cpp"
    "deform/filters/gaussian_filter.h"
    "deform/filters/resample.cpp"
    "deform/filters/resample.h"

    "deform/graph_cut/graph_cut.h"
    "deform/graph_cut/graph_cut.inl"

    "deform/graph_cut/graph_cut.h"
    "deform/graph_cut/graph_cut.inl"
    
    "deform/platform/file_path.cpp"
    "deform/platform/file_path.h"
    "deform/platform/gcc.h"
    "deform/platform/timer_posix.cpp"
    "deform/platform/timer_win.cpp"
    "deform/platform/timer.h"
    "deform/platform/windows_wrapper.h"

    "deform/registration/block_change_flags.cpp"
    "deform/registration/block_change_flags.h"
    "deform/registration/blocked_graph_cut_optimizer.h"
    "deform/registration/blocked_graph_cut_optimizer.inl"
    "deform/registration/registration_engine.cpp"
    "deform/registration/registration_engine.h"
    "deform/registration/settings.cpp"
    "deform/registration/settings.h"
    "deform/registration/transform.cpp"
    "deform/registration/transform.h"
    "deform/registration/volume_pyramid.cpp"
    "deform/registration/volume_pyramid.h"
    "deform/registration/voxel_constraints.cpp"
    "deform/registration/voxel_constraints.h"

    "deform/benchmark.cpp"
    "deform/config.h"
    "deform/cost_function.h"
    "deform/jacobian.cpp"
    "deform/jacobian.h"
    "deform/main.cpp"
    "deform/regularize.cpp"
)

add_executable(deform ${DEFORM_SRCS})

target_include_directories(deform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_link_libraries(deform stk json gco)

