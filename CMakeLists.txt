cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(flexfringe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(COMPILE_DOCS "This is settable from the command line" OFF)

add_subdirectory(source/utility/lexy)
add_subdirectory(source/utility/fmt-9.1.0)

if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
else()
    add_compile_options("-Wall" "-Wextra"
                        "$<$<CONFIG:RELEASE>:-O3>"
                        "$<$<CONFIG:DEBUG>:-g>")
endif()

add_compile_definitions(LOGURU_WITH_STREAMS=1)
add_compile_definitions(CATCH_CONFIG_EXPERIMENTAL_REDIRECT=1)

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
        Util
        foonathan::lexy
        fmt::fmt)


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
            tests/smoketest.cpp
            tests/testcsvheaderparser.cpp
            tests/testcsvparser.cpp
            tests/testabbadingoparser.cpp
            source/main.cpp tests/testinputdata.cpp)

    if(MSVC)
        target_link_libraries(runtests
                Evaluation)
        set_target_properties(runtests PROPERTIES LINK_FLAGS
                "/WHOLEARCHIVE:Evaluation")
    elseif(APPLE)
        target_link_libraries(runtests
                "-Wl,-force_load" Evaluation
                "-Wl")
    else()
        target_link_libraries(runtests
                "-Wl,--whole-archive" Evaluation
                "-Wl,--no-whole-archive")
    endif()

    target_link_libraries(runtests
            Catch2::Catch2
            Source
            Util
            foonathan::lexy
            fmt::fmt
            ${CMAKE_THREAD_LIBS_INIT})

    if(NOT WIN32)
        target_link_libraries(runtests dl) # For ldl
    endif()

    target_compile_definitions(runtests PUBLIC
            UNIT_TESTING)
endif()

if(MSVC)
    # Statically link msvc runtime so we don't need the redistributable
    set_property(TARGET flexfringe Evaluation Source Util runtests PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
