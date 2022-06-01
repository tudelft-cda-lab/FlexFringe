cmake_minimum_required(VERSION 3.16)
project(flexfringe)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(COMPILE_DOCS "This is settable from the command line" OFF)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_compile_definitions(LOGURU_WITH_STREAMS=1)

include_directories("${PROJECT_SOURCE_DIR}"
                    "${PROJECT_SOURCE_DIR}/source"
                    "${PROJECT_SOURCE_DIR}/source/evaluation"
                    "${PROJECT_SOURCE_DIR}/source/utility")

add_subdirectory(source)

add_executable(flexfringe source/main.cpp)

if(MSVC)
    target_link_libraries(flexfringe
            Evaluation)
    # MSVC option to add all object files in static lib
    set_target_properties(flexfringe PROPERTIES LINK_FLAGS
            "/WHOLEARCHIVE:Evaluation")

elseif(APPLE)
	target_link_libraries(flexfringe
            "-Wl,-force_load" Evaluation
            "-Wl")

else()
    # Need this so the plugin registration does not get stripped out
    target_link_libraries(flexfringe
            "-Wl,--whole-archive" Evaluation
            "-Wl,--no-whole-archive")
endif()

target_link_libraries(flexfringe
        Source
        Util)


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

    add_executable(runtests
            tests/tests.cpp
            tests/tail.cpp)

    target_link_libraries(runtests PRIVATE
            Catch2::Catch2
            Evaluation
            Source
            Util)
endif()

if(MSVC)
    # Statically link msvc runtime so we don't need the redistributable
    set_property(TARGET flexfringe Evaluation Source Util runtests PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()