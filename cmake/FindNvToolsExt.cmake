include(FindPackageHandleStandardArgs)

if (WIN32)
    find_path(NvToolsExt_INCLUDE_DIR
        NAMES
            nvToolsExt.h
        PATH_SUFFIXES
            include
        PATHS
            $ENV{NvToolsExt_PATH})

    find_library(NvToolsExt_LIBRARY_RELEASE "nvToolsExt64_1"
        PATH_SUFFIXES
            "lib/x64" "lib"
        PATHS
            $ENV{NvToolsExt_PATH})

    find_library(NvToolsExt_LIBRARY_DEBUG "nvToolsExt64_1"
        PATH_SUFFIXES
            "lib/x64" "lib"
        PATHS
            $ENV{NvToolsExt_PATH})

else ()
    find_path(NvToolsExt_INCLUDE_DIR
        NAMES
            nvToolsExt.h
        PATH_SUFFIXES
            include
        PATHS
            "/opt/cuda/")

    find_library(NvToolsExt_LIBRARY_RELEASE "libnvToolsExt.so"
        PATH_SUFFIXES
            "lib64/"
        PATHS
            "/opt/cuda/")

    find_library(NvToolsExt_LIBRARY_DEBUG "libnvToolsExt.so"
        PATH_SUFFIXES
            "lib64/"
        PATHS
            "/opt/cuda/")

endif ()

include(SelectLibraryConfigurations)
select_library_configurations(NvToolsExt)

find_package_handle_standard_args(
    NvToolsExt 
    DEFAULT_MSG 
    NvToolsExt_INCLUDE_DIR
    NvToolsExt_LIBRARY)

if (NvToolsExt_FOUND)
    set(NvToolsExt_INCLUDE_DIRS ${NvToolsExt_INCLUDE_DIR})
    set(NvToolsExt_LIBRARIES "${NvToolsExt_LIBRARY}")
endif()

mark_as_advanced(NvToolsExt_LIBRARIES_INCLUDE_DIRS NvToolsExt_LIBRARIES_LIBRARIES)