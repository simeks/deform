if(ACTION STREQUAL FETCH)
    find_package(Git)

    if (GIT_FOUND)
        execute_process(COMMAND
            "${GIT_EXECUTABLE}" log -1 --pretty=format:%H
            WORKING_DIRECTORY "${WORKING_DIR}"
            OUTPUT_VARIABLE GIT_SHA1
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        execute_process(COMMAND
            "${GIT_EXECUTABLE}" log -1 --pretty=format:%h
            WORKING_DIRECTORY "${WORKING_DIR}"
            OUTPUT_VARIABLE GIT_SHA1_SHORT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        execute_process(COMMAND
            "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY "${WORKING_DIR}"
            OUTPUT_VARIABLE GIT_BRANCH
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        execute_process(COMMAND
            "${GIT_EXECUTABLE}" status --porcelain
            WORKING_DIRECTORY "${WORKING_DIR}"
            OUTPUT_VARIABLE GIT_STATUS
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if (NOT "${GIT_STATUS}" STREQUAL "")
            set(GIT_DIRTY "true")
        else()
            set(GIT_DIRTY "false")
        endif()

        execute_process(COMMAND
            "${GIT_EXECUTABLE}" describe --tags
            WORKING_DIRECTORY "${WORKING_DIR}"
            OUTPUT_VARIABLE GIT_TAG_STR
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        string(REGEX REPLACE "^v(.*)-.*-.*" "\\1" GIT_VERSION_TAG "${GIT_TAG_STR}")

    else()
        set(GIT_SHA1 "Not found")
        set(GIT_SHA1_SHORT "Not found")
        set(GIT_BRANCH "")
        set(GIT_DIRTY "")
        set(GIT_VERSION_TAG "")
    endif()

    configure_file("${DF_VERSION_FILE_IN}" "${DF_VERSION_FILE_OUT}")
else()
    add_custom_target(FetchGitVersion
        DEPENDS ${DF_VERSION_FILE_IN}
        BYPRODUCTS ${DF_VERSION_FILE_OUT}
        COMMAND
            ${CMAKE_COMMAND}
            -DACTION=FETCH
            -DWORKING_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DDF_VERSION_FILE_IN=${DF_VERSION_FILE_IN}
            -DDF_VERSION_FILE_OUT=${DF_VERSION_FILE_OUT}
            -P "${CMAKE_CURRENT_LIST_FILE}"
    )
endif()