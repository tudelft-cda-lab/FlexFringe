cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(flexfringe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

option(COMPILE_DOCS "This is settable from the command line" OFF)
option(ENABLE_DATABASE "Set this for the database connector to work" OFF)
option(ENABLE_PYTHON "Set this for python connector to work" OFF)

if (ENABLE_DATABASE)
  add_definitions(-D__FLEXFRINGE_DATABASE)
endif()
if (ENABLE_PYTHON)
  add_definitions(-D__FLEXFRINGE_PYTHON)
endif()

#set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

# Default compiler flags:
# Set type with: -DCMAKE_BUILD_TYPE=Debug
# Default is Release, set in line 9 of this file.
# Release:
# -O3 (Be fast)
# -DNDEBUG (Disable assert)
# Debug:
# -g (Debug symbols)
# https://stackoverflow.com/a/24767451/8477066
# Extra:
# -D_GLIBCXX_DEBUG Use safe STD (with bounds checking etc).
# -ggdb3 Add extra information for gdb debugger.
# -g Also for Release, it only makes binary bigger and is still useful in practise.
# -UDEBUG Removes #ifdef DEBUG SOME_DEBUG_CODE #endif for your own debug code.
if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      "$<$<CONFIG:RELEASE>:-g;-UDEBUG>"
      "$<$<CONFIG:DEBUG>:-ggdb3;-D_GLIBCXX_DEBUG>")
else()
    add_compile_options(
      "$<$<CONFIG:RELEASE>:-g;-UDEBUG>"
      "$<$<CONFIG:DEBUG>:-ggdb3;-D_GLIBCXX_DEBUG>"
      "-fcompare-debug-second") # This flag is only for me, to suppress notes
endif()

add_compile_definitions(LOGURU_WITH_STREAMS=1)
#add_compile_definitions(CATCH_CONFIG_EXPERIMENTAL_REDIRECT=1)

include(FetchContent)
function(CloneRepository repositoryURL gitTag projectName sourceDir)
  #Commands are left empty so that we only checkout the source and no not perform any kind of build
  message("Starting to clone ${projectName} into ${sourceDir}")
  FetchContent_Declare(
    "${projectName}"
    GIT_REPOSITORY "${repositoryURL}"
    SOURCE_DIR "${sourceDir}"
    GIT_TAG "${gitTag}"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
  )
  FetchContent_MakeAvailable(${projectName})
endfunction(CloneRepository)

if (ENABLE_DATABASE)
  # On Windows we generally recommend building libpqxx as a shared
  # library.  On other platforms, we recommend a static library.
  # You can usually skip building the libpqxx tests.
  set(SKIP_BUILD_TEST ON)
  IF (WIN32)
      set(BUILD_SHARED_LIBS ON)
  ELSE()
      set(BUILD_SHARED_LIBS OFF)
  ENDIF()

  # Please note that libpqxx requires pre-building its library
  # you cannot rely on including header files globally
  # you must link with target_link_libraries(.. lipqxx::pqxx) in EVERY module that requires it.
  # git tag from version 7.9.0.
  CloneRepository("https://github.com/jtv/libpqxx" "2ffb711" "libpqxx" "build-pqxx")
endif()

# Boost is really nice library. Much better than std::regex and as many more nice components.
# set(BOOST_INCLUDE_LIBRARIES asio regex algorithm)
# set(BOOST_ENABLE_CMAKE ON)
# FetchContent_Declare(
#   Boost
#   URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
#   URL_MD5 893b5203b862eb9bbd08553e24ff146a
# )
# FetchContent_MakeAvailable(Boost)


add_subdirectory(source/utility/lexy)
add_subdirectory(source/utility/fmt-11.0.2)

# needed for the Neural Network, to call Python scripts
if (ENABLE_PYTHON)
  # Automatic path setting does not work on Apple Silicon
  if(APPLE)
    message("Getting PythonLibs on Apple (MacOSX) path")
    set(CMAKE_FIND_FRAMEWORK NEVER)
    find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
    
    message("Python_VERSION:${Python_VERSION}")
    message("Python_LIBRARIES:${Python_LIBRARIES}")
    message("Python_INCLUDE_DIRS:${${Python_INCLUDE_DIRS}}")

    set(PYTHON_LIBRARIES ${Python_LIBRARIES})
    string(COMPARE EQUAL "${PYTHON_LIBRARIES}" "" PYTHONLIBS_EMPTY)
    if(PYTHONLIBS_EMPTY)
      message(FATAL_ERROR "Problem: PYTHON_LIBRARIES not found. Do you have Python installed on your machine?")
    endif()

    string(COMPARE EQUAL "${Python_INCLUDE_DIRS}" "" PYTHONDIRS_EMPTY)
    if(PYTHONDIRS_EMPTY)
      message("Failed to automatically find Python_INCLUDE_DIRS. Setting the PYTHON_INCLUDE_DIRS variable manually. If this crashes please adjust the following
              path to the path where Python.h resides (the one matching the found Python instance). Paths must be consistent iff multiple Python versions on machine.")
      set(PYTHON_H_PATH "/usr/local/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13")
      set(PYTHON_INCLUDE_DIRS "${PYTHON_H_PATH}")
    else()
      set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    endif()
    #FindPython3()
    message("Apple - Using Python:${Python_VERSION_MAJOR} - Libraries:${PYTHON_LIBRARIES} - IncludeDirs: ${PYTHON_INCLUDE_DIRS}")
  else()
    message("Getting PythonLibs on Linux or Windows path")
    find_package(PythonLibs REQUIRED)
  endif()
  include_directories(${PYTHON_INCLUDE_DIRS})
  message("Using Python:${Python_VERSION_MAJOR} - Libraries:${PYTHON_LIBRARIES} - IncludeDirs: ${PYTHON_INCLUDE_DIRS}")
endif()
include_directories("${PROJECT_SOURCE_DIR}"
                    "${PROJECT_SOURCE_DIR}/source"
                    "${PROJECT_SOURCE_DIR}/source/evaluation"
                    "${PROJECT_SOURCE_DIR}/source/running_modes"
                    "${PROJECT_SOURCE_DIR}/source/utility"
                    "${PROJECT_SOURCE_DIR}/source/utility/lexy/include"
                    "${PROJECT_SOURCE_DIR}/source/utility/fmt-11.0.2/include")

add_subdirectory(source)

add_executable(flexfringe source/main.cpp)


# Need this so the plugin registration does not get stripped out
if(MSVC)
    target_link_libraries(flexfringe
            Evaluation
            ActiveLearning
    )
    # MSVC option to add all object files in static lib
    set_target_properties(flexfringe PROPERTIES LINK_FLAGS
            "/WHOLEARCHIVE:Evaluation")

elseif(APPLE)
	target_link_libraries(flexfringe
            "-Wl,-force_load" Evaluation
            "-Wl,-force_load" ActiveLearning
            "-Wl")
else()
    target_link_libraries(flexfringe
            "-Wl,--whole-archive" Evaluation
            "-Wl,--whole-archive" ActiveLearning
            "-Wl,--no-whole-archive")
endif()


target_link_libraries(flexfringe
        Source
        Util
        lexy
        fmt
        )
if (ENABLE_DATABASE)
  target_link_libraries(flexfringe libpqxx::pqxx)
endif()


find_package(Threads)
target_link_libraries(flexfringe ${CMAKE_THREAD_LIBS_INIT}) # For pthreads

if(NOT WIN32)
    target_link_libraries(flexfringe dl) # For ldl
endif()

# compile the documentation
if(COMPILE_DOCS)
    # Add the cmake folder so the FindSphinx module for the documentation is found
    set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    add_subdirectory(docs)
endif()

if(NOT SKIP_TESTS)
    add_subdirectory(source/utility/Catch2-2.13.9)

    include_directories(
        "${PROJECT_SOURCE_DIR}/source/active_learning/active_learning_util"
        "${PROJECT_SOURCE_DIR}/source/active_learning/memory"
        "${PROJECT_SOURCE_DIR}/source/active_learning/system_under_learning"
        "${PROJECT_SOURCE_DIR}/source/active_learning/counterexample_strategies"
        "${PROJECT_SOURCE_DIR}/source/active_learning/oracle"
        "${PROJECT_SOURCE_DIR}/source/active_learning/algorithms"
        )

    add_executable(runtests
            tests/tests.cpp
            tests/smoketest.cpp
            tests/testcsvheaderparser.cpp
            tests/testcsvparser.cpp
            tests/testabbadingoparser.cpp
            source/main.cpp
            tests/testinputdata.cpp
            tests/testobservationtable.cpp
            tests/testregex.cpp
            source/main.cpp
            tests/testinputdata.cpp
            )

    # Need this so the plugin registration does not get stripped out
    if(MSVC)
        target_link_libraries(runtests
                Evaluation
                ActiveLearning)
        set_target_properties(runtests PROPERTIES LINK_FLAGS
                "/WHOLEARCHIVE:Evaluation")
    elseif(APPLE)
        target_link_libraries(runtests
                "-Wl,-force_load" Evaluation
                "-Wl,-force_load" ActiveLearning
                "-Wl")
    else()
        target_link_libraries(runtests
                "-Wl,--whole-archive" Evaluation
                "-Wl,--whole-archive" ActiveLearning
                "-Wl,--no-whole-archive")
    endif()


    target_link_libraries(runtests
            Catch2::Catch2
            Source
            Util
            lexy
            fmt
            ${CMAKE_THREAD_LIBS_INIT}
            )
    if (ENABLE_DATABASE)
      target_link_libraries(runtests libpqxx::pqxx)
    endif()
    if(NOT WIN32)
        target_link_libraries(runtests dl) # For ldl
    endif()

    target_compile_definitions(runtests PUBLIC
            UNIT_TESTING)
endif()

# FROM Hielke 2024 October
# I got:
# fmtd.lib(format.obj) : error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MDd_DynamicDebug' doesn't match value 'MTd_StaticDebug' in main.obj [C:\Users\Administrator\repos\FlexFringe\bu ild\flexfringe.vcxproj]
# when this is does not contain all libraries added.
if(MSVC)
    # Statically link msvc runtime so we don't need the redistributable
    set_property(TARGET flexfringe Evaluation Source Util ActiveLearning runtests fmt lexy PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# XCode 15 currently has known issues with LTO:
# https://developer.apple.com/documentation/xcode-release-notes/xcode-15-release-notes#Known-Issues
if (NOT APPLE)
    set_property(TARGET flexfringe PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET runtests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set_property(TARGET flexfringe PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET runtests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
