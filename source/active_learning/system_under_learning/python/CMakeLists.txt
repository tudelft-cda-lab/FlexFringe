cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(flexfringe)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

option(COMPILE_DOCS "This is settable from the command line" OFF)
#set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options(
                        "$<$<CONFIG:RELEASE>:-O3>"
                        "$<$<CONFIG:DEBUG>:-g>"
                        "-fcompare-debug-second") # This flag is only for me, to suppress notes
endif()

# ------------------------------------ Start including python, my own -------------------------------------------

#file(GLOB PYTHON_SEARCH_PATHS "/usr/include/python*" "/home/*/anaconda3/include/python*")
#find_path(
#  PYTHON_INCLUDE_PATH
#  NAMES "Python.h"
#  PATHS ${PYTHON_SEARCH_PATHS}
#)
#
#include_directories(
#  ${PYTHON_INCLUDE_PATH}
#)

#message("Include path: " ${PYTHON_INCLUDE_PATH})

# ------------------------------------- End of including python, my own --------------------------------------------

# 1
find_package(PythonLibs REQUIRED)

add_compile_definitions(LOGURU_WITH_STREAMS=1)

include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(test ./test_python.cpp)
target_link_libraries(test ${PYTHON_LIBRARIES})