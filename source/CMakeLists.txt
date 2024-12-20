include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation"
        "${CMAKE_CURRENT_SOURCE_DIR}/running_modes"
        "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/probabilistic_heuristics"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        "${CMAKE_CURRENT_SOURCE_DIR}/input"
        "${CMAKE_CURRENT_SOURCE_DIR}/input/parsers"
        "${CMAKE_CURRENT_SOURCE_DIR}/input/parsers/grammar"
        "${CMAKE_CURRENT_SOURCE_DIR}/active_learning"
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

file(GLOB ProbFiles "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/probabilistic_heuristics/*.h")
foreach (Filepath ${ProbFiles})
    get_filename_component(Filename ${Filepath} NAME)
    file(APPEND "evaluators.h" "#include \"probabilistic_heuristics/${Filename}\"\n")
endforeach ()
file(APPEND "evaluators.h" "\n#endif")

add_library(Source STATIC
        apta.cpp
        apta.h
        common.h
        common.cpp
        conflict_graph.cpp
        conflict_graph.h
        evaluation_factory.cpp
        evaluation_factory.h
        evaluators.h
        gitversion.cpp
        parameters.cpp
        parameters.h
        refinement.cpp
        refinement.h
        state_merger.cpp
        state_merger.h
        mem_store.cpp
        mem_store.h
        dfa_properties.cpp 
        dfa_properties.h
        refinement_selection_strategies.h 
        refinement_selection_strategies.cpp
        regex_builder.cpp
        regex_builder.h 

        running_mode_factory.h
        running_mode_factory.cpp
        running_modes/dfasat_mode.h
        running_modes/dfasat_mode.cpp
        running_modes/differencing_mode.h
        running_modes/differencing_mode.cpp
        running_modes/ensemble_mode.h
        running_modes/ensemble_mode.cpp
        running_modes/greedy_mode.h
        running_modes/greedy_mode.cpp
        running_modes/interactive_mode.h
        running_modes/interactive_mode.cpp
        running_modes/predict_mode.h
        running_modes/predict_mode.cpp
        running_modes/regex_mode.h
        running_modes/regex_mode.cpp
        running_modes/running_mode_base.h
        running_modes/running_mode_base.cpp
        running_modes/search_mode.h
        running_modes/search_mode.cpp
        running_modes/stream_mode.h
        running_modes/stream_mode.cpp
        running_modes/subgraphextraction_mode.h
        running_modes/subgraphextraction_mode.cpp

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
        input/parsers/reader_strategy.h

        misc/printutil.h
        misc/utils.h
        misc/trim.h
        misc/zip.h
        misc/sqldb.h
        misc/sqldb.cpp
        )

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

file(GLOB CPPProbFiles "${CMAKE_CURRENT_SOURCE_DIR}/evaluation/probabilistic_heuristics/*.h")
foreach (Filepath ${CPPProbFiles})
    get_filename_component(Filename ${Filepath} NAME)
    get_filename_component(Filename ${Filepath} NAME)
    string(REGEX REPLACE "\\.[^.]*$" "" CleanFilename ${Filename})
    file(APPEND "${cmakelists_eval}" "\n            probabilistic_heuristics/${CleanFilename}.h")
    file(APPEND "${cmakelists_eval}" "\n            probabilistic_heuristics/${CleanFilename}.cpp")
endforeach ()

file(APPEND "${cmakelists_eval}" ")")

add_subdirectory(evaluation)
add_subdirectory(utility)
add_subdirectory(active_learning)

# Please note that pqxx should be linked to every module that depends on it.
# include_directories does not work.
if (ENABLE_DATABASE)
        target_link_libraries(Source libpqxx::pqxx)
endif()
