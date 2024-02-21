cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(flexfringe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

option(COMPILE_DOCS "This is settable from the command line" OFF)
#set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

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

# Please note that libpqxx requires pre-building its library
# you cannot rely on including header files globally
# you must link with target_link_libraries(.. pqxx) in EVERY module that requires it.
# git tag from jan 1 2024.
CloneRepository("https://github.com/jtv/libpqxx" "d8b6b9977a0c016fc168186b5fcf695d1af28a52" "libpqxx" "build-pqxx")

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
add_subdirectory(source/utility/fmt-9.1.0)

if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(
                        "$<$<CONFIG:RELEASE>:-O3>"
                        "$<$<CONFIG:DEBUG>:-g>")
else()
    add_compile_options(
                        "$<$<CONFIG:RELEASE>:-O3>"
                        "$<$<CONFIG:DEBUG>:-g>"
                        "-fcompare-debug-second") # This flag is only for me, to suppress notes
endif()

add_compile_definitions(LOGURU_WITH_STREAMS=1)
#add_compile_definitions(CATCH_CONFIG_EXPERIMENTAL_REDIRECT=1)

# needed for the Neural Network, to call Python scripts
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

include_directories("${PROJECT_SOURCE_DIR}"
                    "${PROJECT_SOURCE_DIR}/source"
                    "${PROJECT_SOURCE_DIR}/source/evaluation"
                    "${PROJECT_SOURCE_DIR}/source/utility"
                    "${PROJECT_SOURCE_DIR}/source/utility/lexy/include"
                    "${PROJECT_SOURCE_DIR}/source/utility/fmt-9.1.0/include")

add_subdirectory(source)

add_executable(flexfringe source/main.cpp)

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
    # Need this so the plugin registration does not get stripped out
    target_link_libraries(flexfringe
            "-Wl,--whole-archive" Evaluation
            "-Wl,--whole-archive" ActiveLearning
            "-Wl,--no-whole-archive")
endif()

target_link_libraries(flexfringe
        Source
        Util
        foonathan::lexy
        fmt::fmt
        pqxx
        )


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
        "${PROJECT_SOURCE_DIR}/source/active_learning/teacher"
        "${PROJECT_SOURCE_DIR}/source/active_learning/system_under_learning"
        "${PROJECT_SOURCE_DIR}/source/active_learning/counterexample_strategies"
        "${PROJECT_SOURCE_DIR}/source/active_learning/equivalence_oracle"
        "${PROJECT_SOURCE_DIR}/source/active_learning/algorithms"
        )

    add_executable(runtests
            tests/tests.cpp
            tests/smoketest.cpp
            tests/testcsvheaderparser.cpp
            tests/testcsvparser.cpp
            tests/testabbadingoparser.cpp
            source/main.cpp tests/testinputdata.cpp
            tests/testobservationtable.cpp
            tests/testregex.cpp
            )


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
            foonathan::lexy
            fmt::fmt
            pqxx
            ${CMAKE_THREAD_LIBS_INIT})

    if(NOT WIN32)
        target_link_libraries(runtests dl) # For ldl
    endif()

    target_compile_definitions(runtests PUBLIC
            UNIT_TESTING)
endif()

if(MSVC)
    # Statically link msvc runtime so we don't need the redistributable
    set_property(TARGET flexfringe Evaluation Source Util ActiveLearning runtests PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set_property(TARGET flexfringe PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
set_property(TARGET runtests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
