include_directories("${CMAKE_CURRENT_SOURCE_DIR}/evaluation"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        )

## create the gitversion.cpp
## check version and append either version or "No commit info available\";"
execute_process(COMMAND "git" "rev-parse" "HEAD"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE Gitversion)
if (Gitversion STREQUAL "")
    set(Gitversion "No commit info available")
    message(${Gitversion})
endif ()
string(REPLACE "\n" "" Gitversion ${Gitversion})
file(WRITE "gitversion.cpp" "const char *gitversion = \"${Gitversion}\";")

## create the evaluators.h file
file(GLOB Files "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*.h")
file(WRITE "evaluators.h" "#ifndef __ALL_HEADERS__ \n#define __ALL_HEADERS__ \n\n")
foreach (Filepath ${Files})
    get_filename_component(Filename ${Filepath} NAME)
    file(APPEND "evaluators.h" "#include \"${Filename}\"\n")
endforeach ()
file(APPEND "evaluators.h" "\n#endif")

add_library(Source STATIC
        apta.cpp
        apta.h
        common.h
        conflict_graph.cpp
        conflict_graph.h
        dfasat.cpp
        dfasat.h
        evaluation_factory.cpp
        evaluation_factory.h
        evaluators.h
        gitversion.cpp
        interactive.cpp
        interactive.h
        parameters.cpp
        parameters.h
        greedy.cpp
        greedy.h
        refinement.cpp
        refinement.h
        searcher.cpp
        searcher.h
        state_merger.cpp
        state_merger.h
        stream.cpp
        stream.h
        mem_store.cpp
        mem_store.h
        predict.cpp predict.h
        differencing.cpp differencing.h
        ensemble.cpp ensemble.h
        dfa_properties.cpp dfa_properties.h
        input/trace.cpp
        input/trace.h
        input/inputdatalocator.cpp
        input/inputdatalocator.h
        input/tail.cpp input/tail.h
        input/attribute.cpp
        input/attribute.h
        input/abbadingoreader.cpp
        input/abbadingoreader.h
        input/parsers/csvparser.h
        input/parsers/csvparser.cpp
        input/parsers/i_parser.h
        input/inputdata.h
        input/inputdata.cpp
        input/parsers/symbol_info.cpp
        input/parsers/symbol_info.h
        input/parsers/abbadingoparser.cpp
        input/parsers/abbadingoparser.h
        input/parsers/grammar/abbadingoheader.h
        input/parsers/grammar/abbadingosymbol.h
        input/parsers/attribute_info.cpp
        input/parsers/attribute_info.h
        input/parsers/grammar/csvheader.h
        input/parsers/dummyparser.cpp
        input/parsers/dummyparser.h
        input/streaming.cpp
        input/streaming.h
        input/parsers/reader_strategy.cpp
        input/parsers/reader_strategy.h)

## create CMakeLists.txt in evaluation directory
set(cmakelists_eval "evaluation/CMakeLists.txt")
file(WRITE "${cmakelists_eval}" "add_library(Evaluation STATIC")
file(GLOB CPPFiles "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/*.cpp")
foreach (Filepath ${Files})
    get_filename_component(Filename ${Filepath} NAME)
    string(REGEX REPLACE "\\.[^.]*$" "" CleanFilename ${Filename})
    file(APPEND "${cmakelists_eval}" "\n            ${CleanFilename}.h")
    file(APPEND "${cmakelists_eval}" "\n            ${CleanFilename}.cpp")
endforeach ()
file(APPEND "${cmakelists_eval}" ")")

add_subdirectory(evaluation)
add_subdirectory(utility)
