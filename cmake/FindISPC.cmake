## ======================================================================== ##
## Copyright 2009-2018 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

# ISPC versions to look for, in decending order (newest first)
SET(ISPC_VERSION_WORKING "1.9.2" "1.9.1")
LIST(GET ISPC_VERSION_WORKING -1 ISPC_VERSION_REQUIRED)

IF (NOT ISPC_EXECUTABLE)
  # try sibling folder as hint for path of ISPC
  IF (APPLE)
    SET(ISPC_DIR_SUFFIX "osx")
  ELSEIF(WIN32)
    SET(ISPC_DIR_SUFFIX "windows")
    IF (MSVC_VERSION LESS 1900)
      MESSAGE(WARNING "MSVC 12 2013 is not supported anymore.")
    ELSE()
      LIST(APPEND ISPC_DIR_SUFFIX "windows-vs2015")
    ENDIF()
  ELSE()
    SET(ISPC_DIR_SUFFIX "linux")
  ENDIF()
  FOREACH(ver ${ISPC_VERSION_WORKING})
    FOREACH(suffix ${ISPC_DIR_SUFFIX})
      LIST(APPEND ISPC_DIR_HINT ${PROJECT_SOURCE_DIR}/../ispc-v${ver}-${suffix})
    ENDFOREACH()
  ENDFOREACH()

  FIND_PROGRAM(ISPC_EXECUTABLE ispc HINTS ${ISPC_DIR_HINT} DOC "Path to the ISPC executable.")
  IF (NOT ISPC_EXECUTABLE)
    MESSAGE("********************************************")
    MESSAGE("Could not find ISPC (looked in PATH and ${ISPC_DIR_HINT})")
    MESSAGE("")
    MESSAGE("This program expects you to have a binary install of ISPC minimum version ${ISPC_VERSION_REQUIRED}, and expects it to be found in 'PATH' or in the sibling directory to where the source are located. Please go to https://ispc.github.io/downloads.html, select the binary release for your particular platform, and unpack it to ${PROJECT_SOURCE_DIR}/../")
    MESSAGE("")
    MESSAGE("If you insist on using your own custom install of ISPC, please make sure that the 'ISPC_EXECUTABLE' variable is properly set in CMake.")
    MESSAGE("********************************************")
    MESSAGE(FATAL_ERROR "Could not find ISPC. Exiting.")
  ELSE()
    MESSAGE(STATUS "Found Intel SPMD Compiler (ISPC): ${ISPC_EXECUTABLE}")
  ENDIF()
ENDIF()

IF(NOT ISPC_VERSION)
  EXECUTE_PROCESS(COMMAND ${ISPC_EXECUTABLE} --version OUTPUT_VARIABLE ISPC_OUTPUT)
  STRING(REGEX MATCH " ([0-9]+[.][0-9]+[.][0-9]+)(dev|knl|rc[0-9])? " DUMMY "${ISPC_OUTPUT}")
  SET(ISPC_VERSION ${CMAKE_MATCH_1})

  IF (ISPC_VERSION VERSION_LESS ISPC_VERSION_REQUIRED)
    MESSAGE(FATAL_ERROR "Need at least version ${ISPC_VERSION_REQUIRED} of Intel SPMD Compiler (ISPC).")
  ENDIF()

  SET(ISPC_VERSION ${ISPC_VERSION} CACHE STRING "ISPC Version")
  MARK_AS_ADVANCED(ISPC_VERSION)
  MARK_AS_ADVANCED(ISPC_EXECUTABLE)
ENDIF()

